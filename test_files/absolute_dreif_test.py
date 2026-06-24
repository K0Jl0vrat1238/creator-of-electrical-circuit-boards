import requests

# Адрес твоего сервера FastAPI
BASE_URL = "http://192.168.0.211:8000/api/v0"

def run_cnc_stress_test():
    print("(🚀) Начинаем тест! Отправляю команду на парковку...")

    # 1. Паркинг в самом начале
    try:
        response = requests.post(f"{BASE_URL}/parking")
        if response.status_code != 200:
            print(f"(❌) Ошибка парковки: {response.json().get('detail')}")
            return
        print(f"(✅) Парковка успешна! Ответ сервера: {response.json()}")
    except Exception as e:
        print(f"(💥) Не удалось связаться с сервером: {e}")
        return

    # Перед циклом выводим Y в -160.0, а X в 0
    print("(📍) Выдвигаемся на стартовую позицию (X: 0, Y: -160)...")
    start_payload = {"x": 0.0, "y": -160.0, "diagonal": True}
    response = requests.post(f"{BASE_URL}/move_absolute", json=start_payload)
    if response.status_code != 200:
        print(
            f"(❌) Не удалось выйти на старт: {response.json().get('detail')}"
        )
        return

    print("(🔥) Стартовая позиция принята. Запускаю цикл на 200 проходов!")

    # 2. Цикл на 200 повторений
    for i in range(1, 201):
        print(f"\n--- Итерация {i} / 200 ---")

        # Шаг вперёд: X +1 см (10мм), Y от -160 едет на +1 см (становится -150)
        move_forward = {"x": 10.0, "y": -150.0, "diagonal": True}
        res_forward = requests.post(
            f"{BASE_URL}/move_absolute", json=move_forward
        )

        if res_forward.status_code != 200:
            print(
                f"(⚠️) Сбой при движении вперёд на итерации {i}: {res_forward.json().get('detail')}"
            )
            break

        # Шаг назад: X возвращается в 0, Y возвращается в -160
        move_back = {"x": 0.0, "y": -160.0, "diagonal": True}
        res_back = requests.post(f"{BASE_URL}/move_absolute", json=move_back)

        if res_back.status_code != 200:
            print(
                f"(⚠️) Сбой при возврате назад на итерации {i}: {res_back.json().get('detail')}"
            )
            break

        print(
            f"(✨) Итерация {i} завершена. Текущая позиция: {res_back.json().get('current_pos')}"
        )

    print("\n(🎉) Тест полностью завершён, Повелитель!")


if __name__ == "__main__":
    run_cnc_stress_test()