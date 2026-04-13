(() => {
  const getTargets = () => Array.from(document.querySelectorAll("article.bd-article section.tex2jax_ignore.mathjax_ignore"));

  const prepareTargets = () => {
    const targets = getTargets();
    for (const target of targets) {
      target.classList.remove("tex2jax_ignore", "mathjax_ignore");
      target.classList.add("mathjax_process");
    }
    return targets;
  };

  const tryTypeset = () => {
    const targets = prepareTargets();
    if (!targets.length) {
      return true;
    }

    const mathJax = window.MathJax;
    if (!mathJax) {
      return false;
    }

    if (typeof mathJax.typesetClear === "function") {
      mathJax.typesetClear(targets);
    }

    if (typeof mathJax.typesetPromise === "function") {
      mathJax.typesetPromise(targets).catch(() => {});
      return true;
    }

    if (typeof mathJax.typeset === "function") {
      mathJax.typeset(targets);
      return true;
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
