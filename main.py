import asyncio
import threading
from multiprocessing import Manager, Process
import simulation
import plotting
from chatgpt import open_realtime_session, process_user_input

async def main():
    # 1) Open persistent WebSocket session
    await open_realtime_session()

    # 2) Start the mood simulation thread
    sim_thread = threading.Thread(target=simulation.update_mood, daemon=True)
    sim_thread.start()

    # 3) Launch live plotting processes
    plot3d_proc = Process(
        target=plotting.live_plot_3d,
        args=(simulation.running_flag, simulation.mood_history)
    )
    plot3d_proc.start()

    plot2d_proc = Process(
        target=plotting.update_2d_plots,
        args=(simulation.running_flag, simulation.mood_history)
    )
    plot2d_proc.start()

    # 4) Enter user‐input loop
    await process_user_input()

    # 5) Cleanup
    sim_thread.join()
    simulation.running_flag.value = False
    plot3d_proc.join()
    plot2d_proc.join()
    print("Simulation ended.")

if __name__ == '__main__':
    # Initialize shared state
    manager = Manager()
    shared_mood_history = manager.list()
    simulation.mood_history = shared_mood_history
    simulation.running_flag = manager.Value('b', True)

    # Run the async main
    asyncio.run(main())