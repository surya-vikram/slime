import os

import ray

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def _explicit_eval_due(rollout_id):
    updates = {int(value) for value in os.getenv("MIXRL_EVAL_UPDATES", "").split(",") if value}
    return rollout_id + 1 in updates


def train(args):
    from slime_plugins.chimera_mixrl.budget import RunBudget
    budget = RunBudget.from_env()
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # A resumed run with insufficient startup budget must not consume a batch.
        if budget.should_stop():
            if rollout_id > args.start_rollout_id:
                # A periodic evaluation may have consumed the remaining budget.
                # Save the already completed boundary before consuming another batch.
                if args.offload_rollout:
                    ray.get(rollout_manager.offload.remote())
                if release_train:
                    actor_model.create()
                actor_model.save_model(rollout_id - 1, force_sync=True)
                if args.use_critic:
                    critic_model.save_model(rollout_id - 1, force_sync=True)
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id - 1))
            print(f'MIXRL_STOP before rollout_id={rollout_id}: budget or stop file; no new batch consumed', flush=True)
            break
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))
            if budget.should_stop():
                print('MIXRL_STOP after baseline evaluation: no optimizer step or batch consumed', flush=True)
                break

        budget.begin_update()
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if isinstance(rollout_data_ref, dict) and rollout_data_ref.get('skip_optimizer'):
            print(f'MIXRL_SKIP rollout_id={rollout_id}: no informative groups; optimizer/scheduler unchanged', flush=True)
            stopping = budget.finish_update()
            if stopping or should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                if args.offload_rollout:
                    ray.get(rollout_manager.offload.remote())
                if release_train:
                    actor_model.create()
                actor_model.save_model(rollout_id, force_sync=True)
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id))
                if args.offload_rollout:
                    ray.get(rollout_manager.onload_weights.remote())
                    # SGLang resumes allocation, not the discarded weight data.
                    # A skipped optimizer step still needs the unchanged actor
                    # weights broadcast after checkpoint-time rollout offload.
                    actor_model.update_weights()
                    ray.get(rollout_manager.onload_kv.remote())
            if (((stopping or rollout_id == args.num_rollout - 1) and args.eval_interval is not None)
                    or _explicit_eval_due(rollout_id)
                    or should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)):
                ray.get(rollout_manager.eval.remote(rollout_id, final=stopping or rollout_id == args.num_rollout - 1))
            if stopping:
                print(f'MIXRL_STOP saved rollout_id={rollout_id}; optimizer unchanged', flush=True)
                break
            continue

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        stopping = budget.finish_update()
        if stopping or release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = stopping or release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if (((stopping or rollout_id == args.num_rollout - 1) and args.eval_interval is not None)
                or _explicit_eval_due(rollout_id)
                or should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)):
            ray.get(rollout_manager.eval.remote(rollout_id, final=stopping or rollout_id == args.num_rollout - 1))
        if stopping:
            print(f'MIXRL_STOP saved rollout_id={rollout_id}; next_rollout_id={rollout_id + 1}', flush=True)
            break

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
