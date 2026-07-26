/* Distil docs — progressive enhancements: copy buttons + "on this page" TOC.
   Vanilla, dependency-free, no external requests. */
(function () {
  "use strict";

  // ── Copy-to-clipboard on every code block ──────────────────────────
  // Shared visually-hidden polite live region: announces the copy result to
  // screen reader users, since the visual button-label swap alone is not
  // reliably announced.
  var copyStatus = document.createElement("span");
  copyStatus.setAttribute("aria-live", "polite");
  copyStatus.setAttribute("role", "status");
  copyStatus.style.cssText = "position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0";
  document.body.appendChild(copyStatus);
  function announceCopy(msg) {
    copyStatus.textContent = "";
    setTimeout(function () { copyStatus.textContent = msg; }, 50);
  }

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
        announceCopy("Copied to clipboard");
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
        try {
          document.execCommand("copy");
          done();
        } catch (e) {
          btn.textContent = "Ctrl-C";
          announceCopy("Copy failed, press Ctrl+C to copy manually");
        }
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
  var seenIds = {};
  var lastH2 = "";
  heads.forEach(function (h) {
    if (!h.id) {
      var base = slug(h.textContent) || "section";
      // Prefix h3 ids with the nearest h2's id so repeated subheadings
      // (e.g. every command's "Flags" section) still get unique, readable ids.
      h.id = (h.tagName === "H3" && lastH2) ? lastH2 + "-" + base : base;
    }
    var n = seenIds[h.id] = (seenIds[h.id] || 0) + 1;
    if (n > 1) h.id = h.id + "-" + n;
    if (h.tagName === "H2") lastH2 = h.id;
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
