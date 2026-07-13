(function () {
  function forceLightTheme() {
    document.documentElement.dataset.mode = "light";
    document.documentElement.dataset.theme = "light";
    document.documentElement.style.colorScheme = "light";
    window.localStorage.setItem("mode", "light");
    window.localStorage.setItem("theme", "light");
  }

  function writeToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var helper = document.createElement("textarea");
      helper.value = text;
      helper.setAttribute("readonly", "");
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      helper.style.left = "-9999px";
      document.body.appendChild(helper);
      helper.select();
      try {
        var ok = document.execCommand("copy");
        document.body.removeChild(helper);
        if (ok) {
          resolve();
        } else {
          reject(new Error("copy command was not accepted"));
        }
      } catch (err) {
        document.body.removeChild(helper);
        reject(err);
      }
    });
  }

  function flashCopyState(button, success) {
    var previous = button.textContent;
    button.textContent = success ? "Copied" : "Failed";
    button.classList.toggle("is-copied", success);
    button.classList.toggle("is-error", !success);
    window.setTimeout(function () {
      button.textContent = previous;
      button.classList.remove("is-copied");
      button.classList.remove("is-error");
    }, 1200);
  }

  function consoleInputHtml(html) {
    var input = [];
    var lines = html.replace(/\r\n/g, "\n").split("\n");
    lines.forEach(function (line) {
      var probe = document.createElement("span");
      probe.innerHTML = line;
      var text = probe.textContent || "";
      var prompt = probe.querySelector(".gp");

      if (prompt || text.indexOf(">>>") === 0 || text.indexOf("...") === 0) {
        input.push(line);
      }
    });
    while (input.length && input[input.length - 1] === "") {
      input.pop();
    }
    return input.join("\n");
  }

  function isConsoleBlock(container, pre) {
    if (container.closest(".doctest") || container.closest(".highlight-pycon")) {
      return true;
    }
    return !!pre.querySelector(".gp");
  }

  function setConsoleOutputVisible(container, pre, visible) {
    container.dataset.simyujOutputVisible = visible ? "1" : "0";
    if (visible) {
      pre.innerHTML = container.dataset.simyujFullHtml;
    } else {
      pre.innerHTML = container.dataset.simyujInputHtml;
    }
  }

  function copyTextFromPre(pre) {
    var clone = pre.cloneNode(true);
    var prompts = clone.querySelectorAll(".gp");
    prompts.forEach(function (prompt) {
      prompt.remove();
    });
    return (clone.innerText || clone.textContent || "").replace(
      /(^|\n)(>>> |\.\.\. )/g,
      "$1"
    );
  }

  function attachCopyButton(container, pre) {
    if (
      !container ||
      !pre ||
      container.dataset.simyujCopyReady === "1" ||
      pre.classList.contains("mermaid") ||
      pre.closest(".mermaid-container")
    ) {
      return;
    }
    container.dataset.simyujCopyReady = "1";
    container.classList.add("simyuj-copy-enabled");

    var controls = document.createElement("div");
    controls.className = "simyuj-copy-controls";
    container.appendChild(controls);

    var consoleToggle = null;
    if (isConsoleBlock(container, pre)) {
      container.dataset.simyujFullHtml = pre.innerHTML;
      container.dataset.simyujInputHtml = consoleInputHtml(pre.innerHTML);
      container.dataset.simyujOutputVisible = "1";

      if (container.dataset.simyujInputHtml !== "") {
        consoleToggle = document.createElement("button");
        consoleToggle.className = "simyuj-copy-mode";
        consoleToggle.type = "button";
        consoleToggle.textContent = ">>>";
        consoleToggle.setAttribute("aria-label", "Toggle console output");
        consoleToggle.setAttribute("aria-pressed", "false");
        consoleToggle.addEventListener("click", function () {
          var visible = container.dataset.simyujOutputVisible !== "1";
          setConsoleOutputVisible(container, pre, visible);
          consoleToggle.setAttribute("aria-pressed", visible ? "false" : "true");
        });
        controls.appendChild(consoleToggle);
      }
    }

    var button = document.createElement("button");
    button.className = "simyuj-copy-btn";
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code block");
    button.addEventListener("click", function () {
      var text = copyTextFromPre(pre);
      writeToClipboard(text.replace(/\n$/, ""))
        .then(function () {
          flashCopyState(button, true);
        })
        .catch(function () {
          flashCopyState(button, false);
        });
    });
    controls.appendChild(button);
  }

  function initCodeCopyButtons() {
    var highlightBlocks = document.querySelectorAll(".bd-content div.highlight");
    highlightBlocks.forEach(function (block) {
      var pre = block.querySelector("pre");
      attachCopyButton(block, pre);
    });

    var plainPreBlocks = document.querySelectorAll(".bd-content pre");
    plainPreBlocks.forEach(function (pre) {
      if (
        pre.closest("div.highlight") ||
        pre.classList.contains("mermaid") ||
        pre.closest(".mermaid-container")
      ) {
        return;
      }
      var wrapper = document.createElement("div");
      wrapper.className = "simyuj-copy-pre-wrap";
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      attachCopyButton(wrapper, pre);
    });
  }

  function init() {
    forceLightTheme();
    initCodeCopyButtons();
  }

  forceLightTheme();
  document.addEventListener("DOMContentLoaded", init);
})();
