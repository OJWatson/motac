(() => {
  const getTargets = () => Array.from(document.querySelectorAll(".math")).filter((target) => target.dataset.mathjaxTypeset !== "done");

  const tryTypeset = () => {
    const targets = getTargets();
    if (!targets.length) {
      return true;
    }

    const mathJax = window.MathJax;
    if (!mathJax) {
      return false;
    }

    for (const target of targets) {
      target.dataset.mathjaxTypeset = "pending";
    }

    if (typeof mathJax.typesetClear === "function") {
      mathJax.typesetClear(targets);
    }

    if (typeof mathJax.typesetPromise === "function") {
      mathJax.typesetPromise(targets)
        .then(() => {
          for (const target of targets) {
            target.dataset.mathjaxTypeset = "done";
          }
        })
        .catch(() => {
          for (const target of targets) {
            delete target.dataset.mathjaxTypeset;
          }
        });
      return true;
    }

    if (typeof mathJax.typeset === "function") {
      try {
        mathJax.typeset(targets);
        for (const target of targets) {
          target.dataset.mathjaxTypeset = "done";
        }
        return true;
      } catch {
        for (const target of targets) {
          delete target.dataset.mathjaxTypeset;
        }
        return false;
      }
    }

    return false;
  };

  const start = () => {
    let attempts = 0;
    const tick = () => {
      attempts += 1;
      if (tryTypeset() || attempts >= 50) {
        return;
      }
      window.setTimeout(tick, 200);
    };
    tick();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }

  window.addEventListener("load", start, { once: true });
})();
