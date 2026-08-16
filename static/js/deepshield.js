(function () {
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next =
        document.documentElement.getAttribute("data-theme") === "light"
          ? "dark"
          : "light";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem("ds-theme", next);
      } catch (e) {}
      paintThemeIcon(next);
    });
  }

  function paintThemeIcon(theme) {
    if (!toggle) return;
    toggle.innerHTML =
      theme === "light"
        ? '<i class="fa-solid fa-moon"></i>'
        : '<i class="fa-solid fa-sun"></i>';
    toggle.setAttribute(
      "aria-label",
      theme === "light" ? "Switch to dark theme" : "Switch to light theme"
    );
  }

  paintThemeIcon(document.documentElement.getAttribute("data-theme") || "dark");

  var navToggle = document.getElementById("nav-toggle");
  var navLinks = document.getElementById("nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      var open = navLinks.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  var revealables = document.querySelectorAll(".reveal");
  if (revealables.length) {
    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add("is-in");
              observer.unobserve(entry.target);
            }
          });
        },
        { rootMargin: "0px 0px -40px 0px", threshold: 0.05 }
      );
      revealables.forEach(function (el) {
        observer.observe(el);
      });
    } else {
      revealables.forEach(function (el) {
        el.classList.add("is-in");
      });
    }
  }

  document.addEventListener("click", function (e) {
    var button = e.target.closest(".info-btn");
    if (!button) return;
    var panel = document.getElementById(button.getAttribute("aria-controls"));
    if (!panel) return;
    var open = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", open ? "false" : "true");
    panel.hidden = open;
  });

  if (document.querySelector("[data-tip]")) {
    var tip = document.createElement("div");
    tip.id = "viz-tip";
    document.body.appendChild(tip);

    document.addEventListener("mouseover", function (e) {
      var target = e.target.closest("[data-tip]");
      if (!target) return;
      tip.textContent = target.getAttribute("data-tip");
      tip.style.opacity = "1";
    });

    document.addEventListener("mousemove", function (e) {
      if (tip.style.opacity !== "1") return;
      var x = e.clientX + 14;
      var y = e.clientY - 36;
      if (x + tip.offsetWidth > window.innerWidth - 10) {
        x = e.clientX - tip.offsetWidth - 14;
      }
      if (y < 8) y = e.clientY + 20;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    });

    document.addEventListener("mouseout", function (e) {
      if (e.target.closest("[data-tip]")) tip.style.opacity = "0";
    });
  }
})();

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  var units = ["B", "KB", "MB", "GB"];
  var i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(i ? 1 : 0) + " " + units[i];
}

function createAnalyser(options) {
  var input = document.getElementById(options.inputId);
  var dropzone = document.getElementById(options.dropzoneId);
  var chip = document.getElementById(options.chipId);
  var chipName = document.getElementById(options.chipNameId);
  var chipSize = document.getElementById(options.chipSizeId);
  var submit = document.getElementById(options.submitId);
  var progress = document.getElementById(options.progressId);
  var bar = document.getElementById(options.barId);
  var stage = document.getElementById(options.stageId);
  var pct = document.getElementById(options.pctId);
  var alertBox = document.getElementById(options.alertId);
  var steps = Array.prototype.slice.call(
    document.querySelectorAll("#" + options.progressId + " .pstep")
  );

  var submitLabel = submit ? submit.innerHTML : "";
  var polling = null;

  function showError(message) {
    if (!alertBox) return;
    alertBox.textContent = message;
    alertBox.classList.add("is-on");
  }

  function clearError() {
    if (alertBox) alertBox.classList.remove("is-on");
  }

  function setBusy(busy) {
    if (!submit) return;
    submit.disabled = busy;
    submit.innerHTML = busy
      ? '<span class="spinner"></span> Analysing…'
      : submitLabel;
  }

  function markSteps(value) {
    steps.forEach(function (el) {
      var at = Number(el.getAttribute("data-at") || 0);
      el.classList.remove("is-active", "is-done");
      if (value >= at + 24) el.classList.add("is-done");
      else if (value >= at) el.classList.add("is-active");
    });
  }

  function setProgress(value, text) {
    if (bar) bar.style.width = Math.max(2, value) + "%";
    if (pct) pct.textContent = Math.round(value) + "%";
    if (stage && text) stage.textContent = text;
    markSteps(value);
  }

  if (dropzone) {
    ["dragenter", "dragover"].forEach(function (name) {
      dropzone.addEventListener(name, function (e) {
        e.preventDefault();
        dropzone.classList.add("is-drag");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      dropzone.addEventListener(name, function (e) {
        e.preventDefault();
        dropzone.classList.remove("is-drag");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
      }
    });
  }

  input.addEventListener("change", function () {
    clearError();
    var file = input.files[0];
    if (!file) {
      chip.classList.remove("is-on");
      if (submit) submit.disabled = true;
      return;
    }
    chipName.textContent = file.name;
    chipSize.textContent = formatBytes(file.size);
    chip.classList.add("is-on");
    if (submit) submit.disabled = false;
    if (options.onSelect) options.onSelect(file);
  });

  function poll(jobId) {
    polling = setInterval(function () {
      fetch("/api/job/" + jobId)
        .then(function (r) {
          return r.json();
        })
        .then(function (job) {
          if (job.error && job.status !== "error") return;
          setProgress(job.progress || 0, job.stage);
          if (job.status === "done") {
            clearInterval(polling);
            setProgress(100, "Complete");
            setBusy(false);
            options.onResult(job.result);
          } else if (job.status === "error") {
            clearInterval(polling);
            progress.classList.remove("is-on");
            setBusy(false);
            showError(job.error || "Analysis failed.");
          }
        })
        .catch(function () {
          clearInterval(polling);
          progress.classList.remove("is-on");
          setBusy(false);
          showError("Lost connection to the analysis service.");
        });
    }, 600);
  }

  submit.addEventListener("click", function () {
    var file = input.files[0];
    if (!file) {
      showError("Choose a file first.");
      return;
    }
    clearError();
    if (options.onStart) options.onStart();
    setBusy(true);
    progress.classList.add("is-on");
    setProgress(3, "Uploading file");

    var payload = new FormData();
    payload.append("file", file);

    fetch(options.endpoint, { method: "POST", body: payload })
      .then(function (r) {
        return r.json().then(function (body) {
          return { ok: r.ok, body: body };
        });
      })
      .then(function (res) {
        if (!res.ok) throw new Error(res.body.error || "Upload rejected.");
        setProgress(6, "Queued for analysis");
        poll(res.body.jobId);
      })
      .catch(function (err) {
        progress.classList.remove("is-on");
        setBusy(false);
        showError(err.message);
      });
  });

  if (submit) submit.disabled = true;
}
