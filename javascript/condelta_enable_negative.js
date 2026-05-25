(function () {
    const NEGATIVE_PROMPTS = [
        {promptId: "txt2img_neg_prompt", cfgId: "txt2img_cfg_scale"},
        {promptId: "img2img_neg_prompt", cfgId: "img2img_cfg_scale"},
        {promptId: "hires_neg_prompt", cfgId: "txt2img_hr_cfg"},
    ];
    const DEDICATED_PROMPT_SELECTOR = [
        "#txt2img_condelta_negative_prompt textarea",
        "#img2img_condelta_negative_prompt textarea",
        "#txt2img_condelta_negative_prompt_hidden textarea",
        "#img2img_condelta_negative_prompt_hidden textarea",
    ].join(", ");
    const DEDICATED_STRENGTH_SELECTOR = "#txt2img_condelta_strength, #img2img_condelta_strength";
    const SEAMLESS_THRESHOLD_SELECTOR = [
        "#txt2img_condelta_activation_cfg_threshold",
        "#img2img_condelta_activation_cfg_threshold",
    ].join(", ");
    let scheduled = false;

    function root() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function readNumericComponentValue(id) {
        const component = root().querySelector(`#${id}`);
        if (!component) return null;

        const inputs = component.querySelectorAll("input, textarea");
        for (const input of inputs) {
            const value = Number.parseFloat(input.value);
            if (Number.isFinite(value)) return value;
        }

        const ariaValue = Number.parseFloat(component.getAttribute("aria-valuenow"));
        return Number.isFinite(ariaValue) ? ariaValue : null;
    }

    function setPromptEnabled(promptId, enabled) {
        const component = root().querySelector(`#${promptId}`);
        const textarea = component?.querySelector("textarea");
        if (!textarea) return;

        textarea.disabled = !enabled;

        if (enabled) {
            textarea.removeAttribute("disabled");
            textarea.removeAttribute("aria-disabled");
            component?.removeAttribute("aria-disabled");
        } else {
            textarea.setAttribute("disabled", "disabled");
            textarea.setAttribute("aria-disabled", "true");
            component?.setAttribute("aria-disabled", "true");
        }
    }

    function isDedicatedPromptMode() {
        const app = root();
        if (app.querySelector(DEDICATED_PROMPT_SELECTOR)) return true;

        return Boolean(app.querySelector(DEDICATED_STRENGTH_SELECTOR) && !app.querySelector(SEAMLESS_THRESHOLD_SELECTOR));
    }

    function syncDedicatedNativePromptState() {
        for (const {promptId, cfgId} of NEGATIVE_PROMPTS) {
            const cfg = readNumericComponentValue(cfgId);
            if (cfg === null) continue;

            setPromptEnabled(promptId, cfg > 1.0);
        }
    }

    function enableSeamlessNativePromptState() {
        for (const {promptId} of NEGATIVE_PROMPTS) {
            setPromptEnabled(promptId, true);
        }
    }

    function syncNegativePromptTextareas() {
        scheduled = false;

        if (isDedicatedPromptMode()) {
            syncDedicatedNativePromptState();
        } else {
            enableSeamlessNativePromptState();
        }
    }

    function scheduleEnable() {
        if (scheduled) return;
        scheduled = true;
        setTimeout(syncNegativePromptTextareas, 0);
    }

    function scheduleOnCfgInput(event) {
        if (!event.target?.closest) return;
        if (event.target.closest("#txt2img_cfg_scale, #img2img_cfg_scale, #txt2img_hr_cfg")) {
            scheduleEnable();
        }
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(scheduleEnable);
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(scheduleEnable);
    }

    document.addEventListener("input", scheduleOnCfgInput, true);
    document.addEventListener("change", scheduleOnCfgInput, true);
})();
