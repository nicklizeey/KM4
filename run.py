import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import sys
import subprocess
import webbrowser
import time
import os
import socket
import threading
import queue
import json
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig


import random
import numpy
import torch
# <<<< ADD THIS SECTION FOR SEEDING >>>>
SEED = 0  # Or any fixed integer you choose
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # For full CUDA reproducibility, you might also need these,
    # though they can impact performance:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
# <<<< END OF SEEDING SECTION >>>>

pause_flag = threading.Event()
selected_action = None

# Thread function to listen for pause request (press 'p')
def listen_for_pause_key(input_queue):
    while True:
        key = input("\n🔴 Press [p] + Enter to pause training and return to menu: ").strip().lower()
        if key == 'p':
            input_queue.put("pause")
            pause_flag.set()
            break

# config_map and choose_config() remain the same
config_map = {
    "0": "launch_tensorboard_only",
    "1": "train",
    "2": "train_CNN",
    "3": "train_Transformer_D_O",
    "4": "train_MLP",
    "5": "train_RNN",
    "6": "train_Transformer",
    "q": "quit_program"
}
# Display interactive menu and handle user input
def choose_config():
    print("=" * 60)
    print("🎯 Welcome to the Modular Grokking Toolkit")
    print("Please select an option:")
    print("0 → Launch TensorBoard (view all previous runs)")
    print("1 → Custom config        (train.yaml)")
    print("2 → CNN                 (train_CNN.yaml)")
    print("3 → Transformer D+O     (train_Transformer_D_O.yaml)")
    print("4 → MLP                 (train_MLP.yaml)")
    print("5 → RNN                 (train_RNN.yaml)")
    print("6 → Transformer         (train_Transformer.yaml)")
    print("q → Quit")
    print("=" * 60)
    choice = input("🔢 Enter a number or 'q': ").strip().lower()
    if choice not in config_map:
        print("❌ Invalid choice.")
        sys.exit(1)
    selected = config_map[choice]
    if selected == "launch_tensorboard_only":
        print("\n✅ You selected to launch TensorBoard only.\n")
    elif selected == "quit_program":
        print("\n👋 Exiting program...\n")
    else:
        print(f"\n✅ You selected [{selected}]. Launching training...\n")
        sys.argv.append(f"--config-name={selected}")
    return selected


# Launch TensorBoard and open browser
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except socket.error:
            return False

def launch_tensorboard(logdir, start_port=6006, max_attempts=5):
    port = -1
    for i in range(max_attempts):
        candidate = start_port + i
        if is_port_available(candidate):
            port = candidate
            break
    if port == -1:
        print("❌ Failed to find available port for TensorBoard.")
        return None, -1

    try:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        elif not os.listdir(logdir):
            print(f"INFO: Log directory '{logdir}' is empty.")

        print("\n🕐 Initializing TensorBoard, please wait 5–10 seconds...")
        tb_command = ['tensorboard', '--logdir', logdir, '--port', str(port)]
        tb_process = subprocess.Popen(tb_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"📡 TensorBoard starting on port {port} (PID: {tb_process.pid})")
        time.sleep(7)
        if tb_process.poll() is None:
            url = f"http://localhost:{port}/"
            try:
                webbrowser.open_new_tab(url)
                print(f"🌐 Opened TensorBoard at {url}")
            except:
                print(f"🔗 Please open TensorBoard manually: {url}")
            return tb_process, port
        else:
            print("⚠️ TensorBoard failed to start.")
            return None, port
    except Exception as e:
        print(f"❌ TensorBoard error: {e}")
        return None, -1

# Hydra main entry for training
@hydra.main(version_base=None, config_path="./config", config_name="train_Transformer_D_O")
def main(cfg: DictConfig):
    tb_process, tb_port = None, -1
    input_queue = queue.Queue()

    run_path = HydraConfig.get().runtime.output_dir
    logdir = os.path.join(get_original_cwd(), "outputs")

    trainer = Trainer(OmegaConf.to_container(cfg), hydra_run_path=run_path)
    listener_thread = threading.Thread(target=listen_for_pause_key, args=(input_queue,), daemon=True)
    listener_thread.start()

    tb_process, tb_port = launch_tensorboard(logdir=logdir)

    model_name = cfg['model']['name']
    good_count = 0

    try:
        for epoch in range(0, trainer.max_epoch):
            if pause_flag.is_set():
                print("\n⏸️ Pause detected. Exiting training loop.")
                break

            # Dispatch training method
            if model_name == 'TransformerDecodeOnly':
                trainer.T_D_train_epoch(epoch)
            elif model_name == 'Transformer':
                trainer.T_train_epoch(epoch)
            elif model_name == 'MLP':
                trainer.MLP_train_epoch(epoch)
            elif model_name == 'CNN':
                trainer.CNN_train_epoch(epoch)
            elif model_name == 'RNN':
                trainer.RNN_train_epoch(epoch)
            elif model_name == 'MNIST_MLP':
                trainer.Image_train_epoch(epoch)
            else:
                print(f"❌ Unknown model name: {model_name}")
                break

            if epoch % trainer.eval_every == 0:
                print(f"[Epoch {epoch}] Train Loss: {trainer.train_loss:.4f}, Valid Loss: {trainer.valid_loss:.4f}, "
                      f"Train Acc: {trainer.train_acc:.4f}, Valid Acc: {trainer.valid_acc:.4f}, "
                      f"Param Norm: {trainer.total_param_norm:<.2f}, "
                      f"good acc {good_count} times.")

            if trainer.model_dir and epoch > 0 and epoch % trainer.save_every == 0:
                torch.save(trainer.model.state_dict(),
                           os.path.join(trainer.model_dir, f"model_epoch_{epoch}.pt"))

            if trainer.valid_acc > trainer.stop_acc:
                if good_count >= trainer.after_reach_epoch:
                    print(f"✅ Stop condition met at epoch {epoch}. Saving final model.")
                    if trainer.model_dir:
                        torch.save(trainer.model.state_dict(),
                                   os.path.join(trainer.model_dir, "model_final.pt"))
                    break
                else:
                    good_count += 1
            else:
                good_count = 0

    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tb_process and tb_process.poll() is None:
            tb_process.terminate()
            try:
                tb_process.wait(timeout=5)
                print("✅ TensorBoard terminated.")
            except subprocess.TimeoutExpired:
                tb_process.kill()
                print("⚠️ TensorBoard force killed.")
        if hasattr(trainer, 'writer') and trainer.writer:
            trainer.writer.flush()
            trainer.writer.close()
            print("✅ SummaryWriter closed.")

# Entry point for repeated menu interaction
if __name__ == '__main__':
    while True:
        pause_flag.clear()
        selected_action = choose_config()

        if selected_action == "quit_program":
            print("👋 Exiting Modular Grokking Toolkit.")
            break

        if selected_action == "launch_tensorboard_only":
            logdir = os.path.join(os.getcwd(), "outputs")
            tb_process, tb_port = launch_tensorboard(logdir=logdir)
            if tb_process:
                try:
                    print("🟢 TensorBoard is running. Press Ctrl+C to stop.")
                    while tb_process.poll() is None:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("⛔ Stopping TensorBoard...")
                    tb_process.terminate()
                    tb_process.wait()
            continue

        main()

        again = input("\n🔁 Press Enter to return to main menu or type 'q' to quit: ").strip().lower()
        if again == 'q':
            print("👋 Exiting Modular Grokking Toolkit.")
            break