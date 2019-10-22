def unwrap_env_info(obj):
    return obj.vector_observations[0], obj.rewards[0], obj.local_done[0]