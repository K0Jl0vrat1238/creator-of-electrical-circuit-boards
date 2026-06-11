import platform
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor

def ping_address(ip):
    """Пингует один IP адрес в зависимости от ОС."""
    param = "-n" if platform.system().lower() == "windows" else "-c"
    # Отправляем всего 1 пакет и ставим минимальный таймаут, чтобы не ждать долго
    command = ["ping", param, "1", "-w", "500", ip]
    try:
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

def ping_local_network():
    """Определяет локальную подсеть и быстро пингует все IP в ней."""
    try:
        # Получаем локальный IP компа
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        # Вычисляем маску /24 (самая частая для дома/офиса)
        # Для полной точности нужен был бы psutil, но обойдемся базой: берем первые 3 октета
        ip_parts = local_ip.split(".")
        base_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}."

        # Генерируем пул IP-адресов от 1 до 254
        ip_list = [f"{base_ip}{i}" for i in range(1, 255)]

        # Запускаем пинг в 50 потоков, чтобы это заняло пару секунд, а не вечность
        with ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(ping_address, ip_list)
    except Exception:
        # Если не удалось определить сеть, просто пропускаем шаг
        pass

def get_ip_by_mac(mac_address):
    # Нормализуем мак для поиска в любом формате
    mac_clean = mac_address.lower().replace("-", "").replace(":", "")

    # Попытка 1: Ищем в текущем кэше
    ip = find_in_arp(mac_clean)
    if ip:
        return ip

    # Попытка 2: Кэш пуст — будим сеть и пингуем всех
    ping_local_network()

    # Попытка 3: Проверяем ARP-таблицу еще раз после обновления
    return find_in_arp(mac_clean)


def find_in_arp(mac_clean):
    """Парсит ARP таблицу в поисках нужного мака."""
    try:
        arp_output = subprocess.check_output(["arp", "-a"]).decode(
            "cp866", errors="ignore"
        )
        # Ищем строчку, где есть IP и наш очищенный MAC
        for line in arp_output.splitlines():
            line_clean = line.lower().replace("-", "").replace(":", "")
            if mac_clean in line_clean:
                # Выдергиваем первый попавшийся IP из этой строки
                match = re.search(
                    r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line
                )
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None

