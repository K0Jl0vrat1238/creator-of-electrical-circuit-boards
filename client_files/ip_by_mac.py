import re
import socket
import subprocess
import time
import platform
from concurrent.futures import ThreadPoolExecutor

def get_all_subnets():
    """Собирает все подсети, к которым подключен компьютер (Wi-Fi, Docker, VirtualBox)."""
    subnets = set()
    
    # 1. Смотрим, какие подсети уже "светятся" в таблице ARP (самый надежный метод)
    try:
        arp_output = subprocess.check_output(["arp", "-a"]).decode("cp866", errors="ignore")
        for match in re.finditer(r"\s+(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\s+", arp_output):
            base = match.group(1) + "."
            # Игнорируем мультикаст и системные сети
            if not base.startswith(("224.", "239.", "255.", "127.")):
                subnets.add(base)
    except Exception:
        pass

    # 2. Дополнительно собираем все IP адреса самого компьютера
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
        for ip in ips:
            base = ".".join(ip.split(".")[:3]) + "."
            if not base.startswith("127."):
                subnets.add(base)
    except Exception:
        pass
        
    return subnets

def _ping_ip(ip):
    """Используем системный ping (ОС 100% отправит ARP-запрос). Черные консоли скрыты."""
    try:
        if platform.system().lower() == "windows":
            # CREATE_NO_WINDOW флаг, чтобы не мерцали черные окошки
            creation_flag = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            subprocess.run(["ping", "-n", "1", "-w", "200", ip], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                           creationflags=creation_flag)
        else:
            subprocess.run(["ping", "-c", "1", "-W", "1", ip], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def ping_local_network():
    """Сканирует ВСЕ найденные подсети в многопоточном режиме."""
    subnets = get_all_subnets()
    ip_list = []
    for base in subnets:
        ip_list.extend([f"{base}{i}" for i in range(1, 255)])

    if not ip_list:
        return

    # Пингуем сотни адресов одновременно (займет около 1-2 секунд)
    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(_ping_ip, ip_list)

def get_ip_by_mac(mac_address):
    mac_clean = mac_address.lower().replace("-", "").replace(":", "")

    # Сначала проверяем кэш
    ip = find_in_arp(mac_clean)
    if ip:
        return ip

    # Если станка нет, будим ВСЕ локальные сети
    for _ in range(3):
        ping_local_network()
        time.sleep(0.5) # Даем винде время обновить ARP-кэш
        ip = find_in_arp(mac_clean)
        if ip:
            return ip
            
    return None

def find_in_arp(mac_clean):
    """Ищет MAC адрес в таблице ARP."""
    try:
        arp_output = subprocess.check_output(["arp", "-a"]).decode("cp866", errors="ignore")
        for line in arp_output.splitlines():
            line_clean = line.lower().replace("-", "").replace(":", "")
            if mac_clean in line_clean:
                match = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None