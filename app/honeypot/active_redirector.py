import subprocess
import platform
import logging
from typing import Any
from app.honeypot.honeypot_manager import HoneypotBackend, HoneypotEvent

logger = logging.getLogger(__name__)

class ActiveHoneypotBackend(HoneypotBackend):
    """
    Actively redirects attacker traffic to a Honeypot using iptables.
    Gracefully degrades to logging if not on Linux or if iptables fails.
    """
    name: str = "active_iptables"
    
    def __init__(self, honeypot_ip: str = "127.0.0.1", honeypot_port: int = 2222):
        self.honeypot_ip = honeypot_ip
        self.honeypot_port = honeypot_port
        self.is_linux = platform.system() == "Linux"

    def redirect(self, event: HoneypotEvent) -> dict[str, Any]:
        if not event.src_ip or event.decision not in ("BLOCK", "REDIRECT_HONEYPOT"):
            return {
                "backend": self.name, 
                "redirected": False, 
                "reason": "Not a redirectable event or missing src_ip"
            }

        attacker_ip = event.src_ip
        dst_port = event.dst_port or 22
        protocol = (event.protocol or "tcp").lower()

        # Construct iptables command
        # iptables -t nat -A PREROUTING -p tcp -s <attacker_ip> --dport <dst_port> -j DNAT --to-destination <honeypot_ip>:<honeypot_port>
        cmd = [
            "sudo", "iptables", "-t", "nat", "-A", "PREROUTING",
            "-p", protocol,
            "-s", attacker_ip,
            "--dport", str(dst_port),
            "-j", "DNAT",
            "--to-destination", f"{self.honeypot_ip}:{self.honeypot_port}"
        ]

        if not self.is_linux:
            # Fallback for Windows/Mac during development
            msg = f"[OS=Windows/Mac] Simulated iptables: {' '.join(cmd)}"
            print(msg)
            return {
                "backend": self.name,
                "redirected": True,
                "simulated": True,
                "command": " ".join(cmd),
                "reason": "Simulated redirect (Not on Linux)"
            }

        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {
                "backend": self.name,
                "redirected": True,
                "simulated": False,
                "command": " ".join(cmd),
                "reason": "Successfully added iptables DNAT rule"
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run iptables: {e.stderr}")
            return {
                "backend": self.name,
                "redirected": False,
                "simulated": False,
                "command": " ".join(cmd),
                "reason": f"iptables error: {e.stderr}"
            }
        except Exception as e:
            logger.error(f"Error executing active redirect: {str(e)}")
            return {
                "backend": self.name,
                "redirected": False,
                "simulated": False,
                "reason": f"Unexpected error: {str(e)}"
            }
