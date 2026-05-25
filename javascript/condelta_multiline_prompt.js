(function () {
    const DEDICATED_PROMPT_SELECTOR = [
        "#txt2img_condelta_negative_prompt textarea",
        "#img2img_condelta_negative_prompt textarea",
    ].join(", ");

    function allowPlainEnterInDedicatedPrompt(event) {
        if (event.key !== "Enter") return;
        if (event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
        if (!event.target?.matches?.(DEDICATED_PROMPT_SELECTOR)) return;

        event.stopImmediatePropagation();
    }

    document.addEventListener("keypress", allowPlainEnterInDedicatedPrompt, true);
})();
