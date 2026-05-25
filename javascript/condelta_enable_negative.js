(function () {
    const NEGATIVE_PROMPT_IDS = ["txt2img_neg_prompt", "img2img_neg_prompt", "hires_neg_prompt"];
    let scheduled = false;

    function root() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function enableNegativePromptTextareas() {
        scheduled = false;

        for (const id of NEGATIVE_PROMPT_IDS) {
            const textarea = root().querySelector(`#${id} textarea`);
            if (!textarea) continue;

            if (textarea.disabled) {
                textarea.disabled = false;
            }

            if (textarea.hasAttribute("disabled")) {
                textarea.removeAttribute("disabled");
            }

            if (textarea.getAttribute("aria-disabled") === "true") {
                textarea.removeAttribute("aria-disabled");
            }
        }
    }

    function scheduleEnable() {
        if (scheduled) return;
        scheduled = true;
        setTimeout(enableNegativePromptTextareas, 0);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(scheduleEnable);
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(scheduleEnable);
    }
})();
