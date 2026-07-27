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
    var target = pre.querySelector("code") || pre;
    var clone = target.cloneNode(true); // strip any pre-existing .copy-btn before reading text
    clone.querySelectorAll(".copy-btn").forEach(function (b) { b.remove(); });
    var original = clone.textContent;
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
      ha.setAttribute("aria-label", "Link to section: " + h.textContent.trim());
      h.appendChild(ha);
    }
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/#/g, "").trim();
    if (h.tagName === "H3") a.className = "lvl-3";
    nav.appendChild(a);
    entries.push({ a: a, h: h });
  });
  content.insertBefore(nav, content.firstChild);

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

/* Tab groups: .tabs > .tab[data-tab] switches .tabpanel[data-panel].
   These are plain toggle buttons (not the ARIA tabs/tablist pattern): each
   button reports its own pressed state via aria-pressed, so they stay
   ordinary Tab-and-Enter-operable buttons with no roving tabindex or
   arrow-key contract to maintain. */
(function () {
  document.querySelectorAll(".tabs").forEach(function (grp) {
    var tabs = grp.querySelectorAll(".tab");
    var panels = grp.querySelectorAll(".tabpanel");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var key = tab.getAttribute("data-tab");
        tabs.forEach(function (t) {
          var active = t === tab;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-pressed", active ? "true" : "false");
        });
        panels.forEach(function (p) { p.classList.toggle("is-active", p.getAttribute("data-panel") === key); });
      });
    });
  });
})();

/* Wrap content tables in a focusable, labeled scroll region so keyboard
   users can reach the horizontal scroll that .table-scroll gets on
   narrower viewports (see site.css). Desktop rendering is unaffected. */
(function () {
  "use strict";
  var container = document.querySelector("main.content, .content");
  if (!container) return;
  var lastHeading = null;
  container.querySelectorAll("h1, h2, h3, h4, table").forEach(function (el) {
    if (el.tagName !== "TABLE") {
      lastHeading = el;
      return;
    }
    if (el.closest(".table-scroll")) return; // already wrapped
    var label = lastHeading ? lastHeading.textContent.replace(/#/g, "").trim() : "";
    var wrap = document.createElement("div");
    wrap.className = "table-scroll";
    wrap.setAttribute("tabindex", "0");
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", label || "Scrollable table");
    el.parentNode.insertBefore(wrap, el);
    wrap.appendChild(el);
  });
})();

/* New-tab warning: every link that opens target=_blank gets a visually hidden
   "(opens in new tab)" note, so screen-reader users get the same heads-up
   sighted users infer from the browser's own tab behavior (WCAG 3.2.5, AAA). */
(function () {
  "use strict";
  document.querySelectorAll('a[target="_blank"]').forEach(function (a) {
    if (a.querySelector(".sr-only")) return;
    var note = document.createElement("span");
    note.className = "sr-only";
    note.textContent = " (opens in new tab)";
    a.appendChild(note);
  });
})();

/* ── ⌘K search across the docs ────────────────────────────────────────
   Twenty-one pages with no way to search them meant finding "how do I set an
   admin token" required already knowing which page it lived on.

   Static index (search-index.json, regenerated by scripts/build_search_index.py),
   fetched lazily on first open. No search service, no dependency, no runtime —
   consistent with the rest of the site, and it works from file:// too.

   The trigger button is injected rather than added to 21 HTML files, so the
   markup stays in one place and a new page gets search for free. */
(function () {
  "use strict";

  var sidebar = document.getElementById("sidebar");
  if (!sidebar) return; // landing page has its own nav — not a docs page

  var idx = null, loading = null, results = [], active = 0;

  function load() {
    if (idx) return Promise.resolve(idx);
    if (loading) return loading;
    loading = fetch("search-index.json")
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (j) { idx = j || []; return idx; })
      .catch(function () { idx = []; return idx; }); // offline/file:// → empty, never throws
    return loading;
  }

  // Token-AND, not phrase-substring. A real query like "admin token" is two
  // words that appear near each other, not that exact string — phrase matching
  // returned zero hits for it, which is how this was caught. Every token must
  // appear somewhere in the page for it to match at all; headings that contain
  // every token rank above the page itself, so you land on the section.
  function search(q) {
    var toks = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!toks.length || !idx) return [];
    function hasAll(hay) {
      for (var i = 0; i < toks.length; i++) if (hay.indexOf(toks[i]) === -1) return false;
      return true;
    }
    var hits = [];
    idx.forEach(function (p) {
      var title = p.t.toLowerCase();
      var page = title + " " + p.b;           // b is pre-lowercased at build time
      if (!hasAll(page)) return;              // page cannot satisfy the query
      if (hasAll(title)) {
        hits.push({ url: p.u, title: p.t, sub: p.d, score: 0 });
      }
      var deep = 0;
      p.h.forEach(function (h) {
        if (deep >= 3) return;                // don't let one page fill the list
        if (hasAll(h.t.toLowerCase())) {
          deep++;
          hits.push({
            url: p.u + (h.a ? "#" + h.a : ""),
            title: h.t,
            sub: p.t,
            score: h.a ? 1 : 2,
          });
        }
      });
      // Matched only in prose: still worth offering, ranked last.
      if (!deep && !hasAll(title)) {
        hits.push({ url: p.u, title: p.t, sub: p.d, score: 4 });
      }
    });
    hits.sort(function (a, b) { return a.score - b.score; });
    return hits.slice(0, 12);
  }

  var overlay = document.createElement("div");
  overlay.className = "sk-overlay";
  overlay.hidden = true;
  overlay.innerHTML =
    '<div class="sk-panel" role="dialog" aria-modal="true" aria-label="Search documentation">' +
    '<input class="sk-input" type="search" autocomplete="off" spellcheck="false" ' +
    'placeholder="Search the docs…" aria-label="Search the docs" ' +
    'aria-controls="sk-results" aria-expanded="false"/>' +
    '<ul class="sk-results" id="sk-results" role="listbox"></ul>' +
    '<div class="sk-foot"><kbd>↑</kbd><kbd>↓</kbd> navigate · <kbd>↵</kbd> open · <kbd>esc</kbd> close</div>' +
    "</div>";
  document.body.appendChild(overlay);

  var input = overlay.querySelector(".sk-input");
  var list = overlay.querySelector(".sk-results");
  var status = document.createElement("span");
  status.className = "sr-only";
  status.setAttribute("aria-live", "polite");
  overlay.appendChild(status);

  var lastFocus = null;

  function render() {
    list.innerHTML = "";
    results.forEach(function (r, i) {
      var li = document.createElement("li");
      li.className = "sk-hit" + (i === active ? " is-active" : "");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", i === active ? "true" : "false");
      li.innerHTML = '<span class="sk-t"></span><span class="sk-s"></span>';
      li.querySelector(".sk-t").textContent = r.title;
      li.querySelector(".sk-s").textContent = r.sub || "";
      li.addEventListener("click", function () { go(i); });
      list.appendChild(li);
    });
    input.setAttribute("aria-expanded", results.length ? "true" : "false");
    status.textContent = input.value.trim()
      ? results.length + (results.length === 1 ? " result" : " results")
      : "";
  }

  function go(i) {
    var r = results[i];
    if (r) window.location.href = r.url;
  }

  function open() {
    lastFocus = document.activeElement;
    overlay.hidden = false;
    input.value = "";
    results = []; active = 0; render();
    input.focus();
    load().then(function () { results = search(input.value); render(); });
  }

  function close() {
    overlay.hidden = true;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  input.addEventListener("input", function () {
    load().then(function () { results = search(input.value); active = 0; render(); });
  });

  overlay.addEventListener("click", function (e) {
    if (e.target === overlay) close();
  });

  overlay.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); if (results.length) { active = (active + 1) % results.length; render(); } }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (results.length) { active = (active - 1 + results.length) % results.length; render(); } }
    else if (e.key === "Enter") { e.preventDefault(); go(active); }
  });

  document.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      overlay.hidden ? open() : close();
    }
  });

  // Visible affordance — a keyboard-only feature is a feature most people never
  // find. Injected at the top of the sidebar nav on every docs page.
  var nav = sidebar.querySelector("nav") || sidebar;
  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sk-trigger";
  btn.innerHTML = '<span>Search docs</span><kbd class="sk-kbd">⌘K</kbd>';
  btn.addEventListener("click", open);
  nav.insertBefore(btn, nav.firstChild);
  if (!/Mac|iPhone|iPad/.test(navigator.platform || "")) {
    btn.querySelector(".sk-kbd").textContent = "Ctrl K";
  }
})();
