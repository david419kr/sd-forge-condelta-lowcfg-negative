(function () {
    const promptSelectors = [
        "#txt2img_condelta_negative_prompt textarea",
        "#img2img_condelta_negative_prompt textarea",
        "#txt2img_condelta_negative_prompt input[type='text']",
        "#img2img_condelta_negative_prompt input[type='text']",
    ];

    function promptAreas() {
        return promptSelectors.flatMap((selector) => Array.from(gradioApp().querySelectorAll(selector)));
    }

    function promptRoot(textArea) {
        return textArea?.closest?.("#txt2img_condelta_negative_prompt, #img2img_condelta_negative_prompt") || null;
    }

    function hasIdentifierPatch(fn, flagName) {
        const seen = new Set();
        const stack = [fn];

        while (stack.length > 0) {
            const current = stack.pop();
            if (typeof current !== "function" || seen.has(current)) continue;
            if (current[flagName]) return true;

            seen.add(current);
            stack.push(current.__condeltaOriginal, current.__aamOriginal);
        }

        return false;
    }

    function inheritIdentifierPatchState(target, source) {
        for (const key of ["__condeltaPatched", "__condeltaOriginal", "__aamPatched", "__aamOriginal"]) {
            if (Object.prototype.hasOwnProperty.call(source, key)) {
                target[key] = source[key];
            }
        }
    }

    function patchTextAreaIdentifier() {
        if (typeof getTextAreaIdentifier !== "function") return false;
        if (hasIdentifierPatch(getTextAreaIdentifier, "__condeltaPatched")) return true;

        const original = getTextAreaIdentifier;
        const patched = function (textArea) {
            const root = promptRoot(textArea);
            if (root?.id === "txt2img_condelta_negative_prompt") return ".txt2img.condelta-negative";
            if (root?.id === "img2img_condelta_negative_prompt") return ".img2img.condelta-negative";
            return original(textArea);
        };

        inheritIdentifierPatchState(patched, original);
        patched.__condeltaPatched = true;
        patched.__condeltaOriginal = original;

        try {
            getTextAreaIdentifier = patched;
            globalThis.getTextAreaIdentifier = patched;
        } catch (error) {
            console.debug("ConDelta: tagcomplete identifier patch failed", error);
            return false;
        }

        return true;
    }

    function nativeNegativeAutocompleteReady() {
        return Boolean(
            gradioApp().querySelector(
                "#txt2img_neg_prompt textarea.autocomplete, #img2img_neg_prompt textarea.autocomplete",
            ),
        );
    }

    function setupAutocomplete() {
        if (!patchTextAreaIdentifier()) return;
        if (typeof addAutocompleteToArea !== "function") return;
        if (!nativeNegativeAutocompleteReady()) return;

        for (const area of promptAreas()) {
            if (!area.classList.contains("autocomplete")) {
                addAutocompleteToArea(area);
            }
        }
    }

    function scheduleSetup() {
        setupAutocomplete();
        setTimeout(setupAutocomplete, 500);
        setTimeout(setupAutocomplete, 1500);
        setTimeout(setupAutocomplete, 3000);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(scheduleSetup);
    } else {
        window.addEventListener("load", scheduleSetup);
    }

    if (typeof onUiUpdate === "function") {
        onUiUpdate(setupAutocomplete);
    }
})();
