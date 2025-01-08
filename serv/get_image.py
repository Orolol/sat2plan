import os
import datetime
import subprocess

def get_images(port: int = 18299, ip: str = "213.173.110.213"):
    # Create directory with current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = f"backup_{timestamp}"
    os.makedirs(new_dir, exist_ok=True)
    
    # Change to the new directory
    os.chdir(new_dir)
    
    # Construct and execute scp command
    scp_command = f"scp -r -P {port} -i ~/.ssh/id_ed25519 root@{ip}:/workspace/sat2plan/images/ ."
    subprocess.run(scp_command, shell=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download images via SCP")
    parser.add_argument("--port", type=int, default=18299, help="SSH port")
    parser.add_argument("--ip", type=str, default="213.173.110.213", help="Server IP address")
    
    args = parser.parse_args()
    get_images(args.port, args.ip)
