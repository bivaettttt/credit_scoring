function getNumber(id) {
    const el = document.getElementById(id);
    const v = el.value;
    return v === "" ? null : Number(v);
}

function showError(msg) {
    const resBox = document.getElementById("result");
    resBox.className = "result result--bad";
    resBox.textContent = msg;
}

async function predict() {
    const threshold = getNumber("threshold") ?? 0.5;

    const inputs = document.querySelectorAll("input[type='number']");
    const payload = { threshold };

  // Валидация диапазона + сбор данных
    for (const el of inputs) {
    const id = el.id;
    const val = getNumber(id);

    if (val === null || Number.isNaN(val)) {
        showError(`Поле "${id}": введите число.`);
        return;
    }

    if (id !== "threshold") {
        const min = el.min !== "" ? Number(el.min) : null;
        const max = el.max !== "" ? Number(el.max) : null;

        if (min !== null && val < min) {
        showError(`Поле "${id}": значение меньше минимального (${min}).`);
        return;
        }
        if (max !== null && val > max) {
        showError(`Поле "${id}": значение больше максимального (${max}).`);
        return;
        }
    }

    payload[id] = val;
    }

    const resBox = document.getElementById("result");
    resBox.className = "result result--muted";
    resBox.textContent = "Рассчитываю прогноз…";

    try {
    const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Ошибка запроса");

    const proba = data.proba_bad;
    const decision = data.decision;

    if (decision === "APPROVE") {
        resBox.className = "result result--ok";
        resBox.innerHTML =
        `<b>Решение:</b> ОДОБРЕНИЕ<br/>` +
        `<b>Вероятность риска (класс 1):</b> ${proba}<br/>` +
        `<b>Порог:</b> ${data.threshold}`;
    } else {
        resBox.className = "result result--bad";
        resBox.innerHTML =
        `<b>Решение:</b> ОТКАЗ<br/>` +
        `<b>Вероятность риска (класс 1):</b> ${proba}<br/>` +
        `<b>Порог:</b> ${data.threshold}`;
    }
    } catch (e) {
    showError(`Ошибка: ${e.message}`);
    }
}

document.getElementById("btnPredict").addEventListener("click", predict);
