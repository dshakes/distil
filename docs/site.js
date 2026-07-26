/* Distil docs — progressive enhancements: copy buttons + "on this page" TOC.
   Vanilla, dependency-free, no external requests. */

/* ── Mobile sidebar toggle: shared by every docs page's ☰ button
   (onclick="toggleSidebar()") ─────────────────────────────────────────
   Keeps #sidebar's "open" class and .sidebar-toggle's aria-expanded in
   sync, closes on outside click or Escape, and returns focus to the
   toggle button on Escape. */
(function () {
  "use strict";

  function getSidebar() { return document.getElementById("sidebar"); }
  function getToggle() { return document.querySelector(".sidebar-toggle"); }

  function setOpen(open) {
    var sb = getSidebar();
    if (!sb) return;
    sb.classList.toggle("open", open);
    var btn = getToggle();
    if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  window.toggleSidebar = function () {
    var sb = getSidebar();
    if (!sb) return;
    setOpen(!sb.classList.contains("open"));
  };

  document.addEventListener("click", function (e) {
    var sb = getSidebar();
    if (sb && sb.classList.contains("open") && !sb.contains(e.target) && !e.target.closest(".sidebar-toggle")) {
      setOpen(false);
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    var sb = getSidebar();
    if (sb && sb.classList.contains("open")) {
      setOpen(false);
      var btn = getToggle();
      if (btn) btn.focus();
    }
  });
})();

(function () {
  "use strict";

  // ── Copy-to-clipboard on every code block ──────────────────────────
  document.querySelectorAll("pre").forEach(function (pre) {
    var original = (pre.querySelector("code") || pre).textContent; // capture before button
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var done = function () {
        btn.textContent = "✓ Copied";
        btn.classList.add("copied");
        setTimeout(function () {
          btn.textContent = "Copy";
          btn.classList.remove("copied");
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(original).then(done).catch(fallback);
      } else {
        fallback();
      }
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = original;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { btn.textContent = "Ctrl-C"; }
        document.body.removeChild(ta);
      }
    });
    pre.appendChild(btn);
  });

  // ── "On this page" right-rail TOC built from the content headings ───
  var content = document.querySelector(".content");
  if (!content) return;
  var heads = content.querySelectorAll("h2, h3");
  if (heads.length < 2) return;

  function slug(t) {
    return t.toLowerCase().trim().replace(/[^\w]+/g, "-").replace(/^-+|-+$/g, "");
  }

  var nav = document.createElement("nav");
  nav.className = "toc";
  nav.setAttribute("aria-label", "On this page");
  nav.innerHTML = '<div class="toc-title">On this page</div>';

  var entries = [];
  heads.forEach(function (h) {
    if (!h.id) h.id = slug(h.textContent) || "section";
    // Clickable "#" ref link on the header itself (deep-link any section).
    if (!h.querySelector(".hanchor")) {
      var ha = document.createElement("a");
      ha.className = "hanchor";
      ha.href = "#" + h.id;
      ha.textContent = "#";
      ha.setAttribute("aria-label", "Link to this section");
      h.appendChild(ha);
    }
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/#/g, "").trim();
    if (h.tagName === "H3") a.className = "lvl-3";
    nav.appendChild(a);
    entries.push({ a: a, h: h });
  });
  document.body.appendChild(nav);

  // ── Scrollspy: highlight the section currently in view ──────────────
  var byId = {};
  entries.forEach(function (e) { byId[e.h.id] = e.a; });
  if ("IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (records) {
      records.forEach(function (rec) {
        if (rec.isIntersecting) {
          entries.forEach(function (e) { e.a.classList.remove("active"); });
          var act = byId[rec.target.id];
          if (act) act.classList.add("active");
        }
      });
    }, { rootMargin: "-80px 0px -68% 0px", threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  }
})();

/* Tab groups: .tabs > .tab[data-tab] switches .tabpanel[data-panel]. */
(function () {
  document.querySelectorAll(".tabs").forEach(function (grp) {
    var tabs = grp.querySelectorAll(".tab");
    var panels = grp.querySelectorAll(".tabpanel");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var key = tab.getAttribute("data-tab");
        tabs.forEach(function (t) { t.classList.toggle("is-active", t === tab); });
        panels.forEach(function (p) { p.classList.toggle("is-active", p.getAttribute("data-panel") === key); });
      });
    });
  });
})();
