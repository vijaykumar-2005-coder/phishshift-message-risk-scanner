const messageInput = document.getElementById("message");
const scanButton = document.getElementById("scanButton");
const charCount = document.getElementById("charCount");
const status = document.getElementById("status");
const results = document.getElementById("results");
const resultTitle = document.getElementById("resultTitle");
const explanation = document.getElementById("explanation");
const riskCues = document.getElementById("riskCues");
const prototypeNote = document.getElementById("prototypeNote");

messageInput.addEventListener("input", () => {
    charCount.textContent = `${messageInput.value.length} / 10000`;
});

scanButton.addEventListener("click", async () => {
    const message = messageInput.value.trim();

    if (!message) {
        status.textContent = "Please enter a message first.";
        return;
    }

    scanButton.disabled = true;
    status.textContent = "Analyzing message...";
    results.hidden = true;

    try {
        const response = await fetch("/scan", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message})
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        resultTitle.textContent = data.result;
        explanation.textContent = data.explanation;
        riskCues.replaceChildren();

        if (data.risk_cues.length === 0) {
            const item = document.createElement("li");
            item.textContent = "No configured risk cues detected.";
            riskCues.appendChild(item);
        } else {
            data.risk_cues.forEach(cue => {
                const item = document.createElement("li");
                item.textContent = cue;
                riskCues.appendChild(item);
            });
        }

        prototypeNote.textContent = data.prototype
            ? "Prototype mode: these results come from simple rules, not a trained ML model."
            : "";

        results.hidden = false;
        status.textContent = "Analysis complete.";
    } catch (error) {
        status.textContent = error.message;
    } finally {
        scanButton.disabled = false;
    }
});