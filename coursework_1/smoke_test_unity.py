# save as smoke_test_unity.py in the same folder as Banana.exe
from pathlib import Path
from unityagents import UnityEnvironment

# exe = Path(__file__).with_name("Banana.exe")  # or the subfolder path
# env = UnityEnvironment(file_name=str(exe), no_graphics=True, worker_id=1, base_port=5005)

exe = Path(__file__).with_name("Banana_Windows_x86_64") / "Banana.exe"
env = UnityEnvironment(file_name=str(exe), no_graphics=True, worker_id=1, base_port=5005)



brain_name = env.brain_names[0]
env_info = env.reset(train_mode=True)[brain_name]
print("State dim:", len(env_info.vector_observations[0]))
print("Action dim:", env.brains[brain_name].vector_action_space_size)
env.close()
print("Smoke test OK")
