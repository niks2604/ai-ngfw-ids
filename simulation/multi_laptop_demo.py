"""
Multi-Laptop Simulation: Attacker Script (Laptop A)
This script simulates normal traffic followed by a sudden lateral movement (brute force) attack.
"""
import time
import socket
import argparse
import random
import threading

def generate_normal_traffic(target_ip, duration):
    print(f"[*] Generating normal HTTP traffic to {target_ip} for {duration} seconds...")
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((target_ip, 80))
                s.sendall(b"GET / HTTP/1.1\r\nHost: " + target_ip.encode() + b"\r\n\r\n")
                s.recv(1024)
        except Exception:
            pass
        time.sleep(random.uniform(0.5, 2.0))
    print("[*] Normal traffic phase complete.")

def generate_attack_traffic(target_ip, port, duration):
    print(f"[!] Initiating ATTACK (Lateral Movement / Brute Force) on {target_ip}:{port} for {duration} seconds!")
    start_time = time.time()
    count = 0
    while time.time() - start_time < duration:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((target_ip, port))
                s.sendall(b"SSH-2.0-OpenSSH_8.2p1\r\n")
                s.recv(1024)
                count += 1
        except Exception:
            pass
        time.sleep(0.01) # High rate
    print(f"[!] Attack phase complete. Sent {count} connections.")

def main():
    parser = argparse.ArgumentParser(description="AI-NGFW Attacker Simulation Script")
    parser.add_argument("--gateway-ip", required=True, help="IP of Laptop B (Gateway)")
    parser.add_argument("--victim-ip", required=True, help="IP of Laptop C (Victim)")
    args = parser.parse_args()

    print("=== CogniShield AI-NGFW Simulation ===")
    
    # Phase 1: Normal Traffic (to Gateway/Victim)
    t1 = threading.Thread(target=generate_normal_traffic, args=(args.victim_ip, 10))
    t1.start()
    t1.join()

    # Phase 2: Lateral Movement (SSH Brute Force to Gateway/Victim)
    t2 = threading.Thread(target=generate_attack_traffic, args=(args.gateway_ip, 2222, 10))
    t2.start()
    t2.join()
    
    print("=== Simulation Complete ===")

if __name__ == "__main__":
    main()
