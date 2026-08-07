(() => {
  "use strict";

  let initialBootstrapToken = consumeBootstrapToken();
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const elements = {
    deviceChip: $("#deviceChip"),
    deviceLabel: $("#deviceLabel"),
    libraryButton: $("#toggleLibraryButton"),
    libraryCount: $("#libraryCount"),
    libraryPanel: $("#libraryPanel"),
    closeLibraryButton: $("#closeLibraryButton"),
    welcomeLibraryButton: $("#welcomeLibraryButton"),
    searchForm: $("#searchForm"),
    searchInput: $("#searchInput"),
    searchSubmit: $(".search-submit"),
    filters: $$("#kindFilters [data-kind]"),
    promptButtons: $$('[data-prompt]'),
    indexForm: $("#indexForm"),
    folderPaths: $("#folderPaths"),
    indexButton: $("#indexButton"),
    segmentCount: $("#segmentCount"),
    assetSummary: $("#assetSummary"),
    photoCount: $("#photoCount"),
    videoCount: $("#videoCount"),
    jobPanel: $("#jobPanel"),
    jobTitle: $("#jobTitle"),
    jobProgressLabel: $("#jobProgressLabel"),
    jobProgressTrack: $("#jobProgressTrack"),
    jobProgressBar: $("#jobProgressBar"),
    jobCurrent: $("#jobCurrent"),
    jobCounts: $("#jobCounts"),
    resultsKicker: $("#resultsKicker"),
    resultsTitle: $("#resultsTitle"),
    resultMeta: $("#resultMeta"),
    welcomeState: $("#welcomeState"),
    resultGrid: $("#resultGrid"),
    loadingGrid: $("#loadingGrid"),
    emptyState: $("#emptyState"),
    emptyMessage: $("#emptyMessage"),
    errorState: $("#errorState"),
    errorTitle: $("#errorTitle"),
    errorMessage: $("#errorMessage"),
    retryButton: $("#retryButton"),
    footerEngine: $("#footerEngine"),
    viewer: $("#mediaViewer"),
    viewerStage: $("#viewerStage"),
    viewerKind: $("#viewerKind"),
    viewerTitle: $("#viewerTitle"),
    viewerPath: $("#viewerPath"),
    viewerMeta: $("#viewerMeta"),
    openMediaButton: $("#openMediaButton"),
    revealMediaButton: $("#revealMediaButton"),
    toastRegion: $("#toastRegion"),
  };

  const state = {
    kind: "all",
    lastQuery: "",
    searchController: null,
    statusTimer: null,
    currentResult: null,
    lastJobId: null,
    lastJobState: null,
    statusFailures: 0,
    statusInFlight: false,
    authorized: null,
    pendingBootstrapToken: initialBootstrapToken,
  };
  initialBootstrapToken = null;

  const terminalJobStates = new Set([
    "complete",
    "completed",
    "done",
    "failed",
    "error",
    "cancelled",
    "canceled",
  ]);

  const videoIcon = `
    <svg aria-hidden="true" viewBox="0 0 16 16">
      <path d="M5 3.8v8.4L12 8 5 3.8Z"></path>
    </svg>`;

  const photoIcon = `
    <svg aria-hidden="true" viewBox="0 0 16 16">
      <path d="M2.2 3.3h11.6v9.4H2.2zM3.5 11l3-3.1 2.1 2 1.8-2 2.1 3.1M5.3 6.1h.1"></path>
    </svg>`;

  function consumeBootstrapToken() {
    const url = new URL(window.location.href);
    const fragment = new URLSearchParams(url.hash.startsWith("#") ? url.hash.slice(1) : url.hash);
    const hadBootstrapFragment = fragment.has("token");
    const token = fragment.get("token")?.trim() || null;
    const hadLegacyQuery = url.searchParams.has("q") || url.searchParams.has("kind");

    if (hadBootstrapFragment) url.hash = "";
    url.searchParams.delete("q");
    url.searchParams.delete("kind");

    if (hadBootstrapFragment || hadLegacyQuery) {
      window.history.replaceState(null, document.title, `${url.pathname}${url.search}${url.hash}`);
    }
    return token;
  }

  function init() {
    bindEvents();
    setShortcutLabel();
    setSessionControls(false);
    openPrivateSession();
  }

  function bindEvents() {
    elements.searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      search(elements.searchInput.value);
    });

    elements.filters.forEach((button) => {
      button.addEventListener("click", () => {
        const nextKind = button.dataset.kind;
        if (nextKind === state.kind) return;
        setKind(nextKind);
        if (elements.searchInput.value.trim()) search(elements.searchInput.value);
      });
    });

    elements.libraryButton.addEventListener("click", () => toggleLibraryPanel());
    elements.closeLibraryButton.addEventListener("click", () => toggleLibraryPanel(false));
    elements.welcomeLibraryButton.addEventListener("click", () => openLibraryPanel(true));

    elements.indexForm.addEventListener("submit", indexFolders);
    elements.retryButton.addEventListener("click", retryCurrentState);

    elements.promptButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const prompt = button.dataset.prompt;
        elements.searchInput.value = prompt;
        search(prompt);
      });
    });

    elements.openMediaButton.addEventListener("click", () => performAssetAction("open"));
    elements.revealMediaButton.addEventListener("click", () => performAssetAction("reveal"));

    elements.viewer.addEventListener("close", clearViewer);
    elements.viewer.addEventListener("click", (event) => {
      if (event.target === elements.viewer) elements.viewer.close();
    });

    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        elements.searchInput.focus();
        elements.searchInput.select();
      }

      if (event.key === "Escape") closeCardMenus();
    });

    document.addEventListener("click", (event) => {
      if (!event.target.closest(".card-menu") && !event.target.closest(".card-more")) {
        closeCardMenus();
      }
    });

    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && state.authorized !== false) {
        clearTimeout(state.statusTimer);
        fetchStatus();
      }
    });
  }

  function setShortcutLabel() {
    const shortcut = $(".search-shortcut");
    if (!shortcut) return;
    if (/Mac|iPhone|iPad/.test(navigator.platform)) shortcut.innerHTML = "⌘ K";
  }

  function setKind(kind) {
    state.kind = kind;
    elements.filters.forEach((button) => {
      const selected = button.dataset.kind === kind;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  function toggleLibraryPanel(force) {
    const shouldOpen = typeof force === "boolean" ? force : elements.libraryPanel.hidden;
    elements.libraryPanel.hidden = !shouldOpen;
    elements.libraryButton.setAttribute("aria-expanded", String(shouldOpen));
    if (!shouldOpen) elements.libraryButton.focus();
  }

  function openLibraryPanel(focusPaths = false) {
    toggleLibraryPanel(true);
    elements.libraryPanel.scrollIntoView({ behavior: "smooth", block: "center" });
    if (focusPaths) window.setTimeout(() => elements.folderPaths.focus(), 400);
  }

  async function apiFetch(url, options = {}) {
    const config = { ...options };
    config.headers = { Accept: "application/json", ...(options.headers || {}) };
    config.cache = options.cache || "no-store";
    config.credentials = options.credentials || "same-origin";

    if (config.body && typeof config.body !== "string") {
      config.body = JSON.stringify(config.body);
      config.headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, config);
    const contentType = response.headers.get("content-type") || "";
    let payload = null;

    if (response.status !== 204) {
      if (contentType.includes("application/json")) {
        payload = await response.json();
      } else {
        const text = await response.text();
        payload = text ? { detail: text } : null;
      }
    }

    if (!response.ok) {
      const error = new Error(readError(payload, `Request failed (${response.status})`));
      error.status = response.status;
      error.payload = payload;
      throw error;
    }

    return payload;
  }

  function readError(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload === "string") return payload;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg || String(item)).join(" · ");
    }
    if (typeof payload.message === "string") return payload.message;
    return fallback;
  }

  async function openPrivateSession() {
    renderSessionPending();
    if (state.pendingBootstrapToken) {
      try {
        await apiFetch("/api/session", {
          method: "POST",
          headers: { "X-Lumen-Bootstrap": state.pendingBootstrapToken },
        });
        state.pendingBootstrapToken = null;
      } catch (error) {
        if (isUnauthorized(error)) {
          state.pendingBootstrapToken = null;
          await fetchStatus();
        } else {
          renderEngineError();
          renderConnectionError();
        }
        return;
      }
    }
    await fetchStatus();
  }

  async function retryCurrentState() {
    if (state.authorized !== true) {
      setButtonBusy(elements.retryButton, true, "Checking…");
      try {
        await openPrivateSession();
      } finally {
        setButtonBusy(elements.retryButton, false);
      }
      return;
    }
    search(state.lastQuery || elements.searchInput.value);
  }

  function renderSessionPending() {
    elements.deviceChip.classList.remove("is-ready", "is-error", "is-fallback");
    elements.deviceChip.classList.add("is-checking");
    elements.deviceLabel.textContent = "Opening private session";
    elements.deviceChip.title = "Establishing private local access";
    elements.footerEngine.textContent = "Opening private session";
  }

  function renderSessionRequired() {
    clearTimeout(state.statusTimer);
    state.authorized = false;
    setSessionControls(false);
    elements.deviceChip.classList.remove("is-checking", "is-ready", "is-fallback");
    elements.deviceChip.classList.add("is-error");
    elements.deviceLabel.textContent = "Private session required";
    elements.deviceChip.title = "This tab does not have a valid Lumen session";
    elements.footerEngine.textContent = "No private session";
    elements.resultsKicker.textContent = "Private local access";
    elements.resultsTitle.textContent = "Open Lumen from its launcher";
    elements.resultMeta.textContent = "";
    elements.errorTitle.textContent = "Private session required";
    elements.errorMessage.textContent =
      "This tab does not have access to your library. Run “uv run lumen serve --open” and use the newly opened tab, then check again here.";
    elements.retryButton.textContent = "Check again";
    showSearchState("error");
  }

  function renderConnectionError() {
    state.authorized = null;
    setSessionControls(false);
    elements.resultsKicker.textContent = "Local service unavailable";
    elements.resultsTitle.textContent = "Lumen is not responding";
    elements.resultMeta.textContent = "";
    elements.errorTitle.textContent = "Couldn’t reach Lumen";
    elements.errorMessage.textContent = "Make sure the local Lumen service is still running, then try again.";
    elements.retryButton.textContent = "Try again";
    showSearchState("error");
  }

  function restoreSessionView() {
    elements.errorTitle.textContent = "Search hit a snag";
    elements.errorMessage.textContent = "Lumen couldn’t reach its local search engine.";
    elements.retryButton.textContent = "Try again";
    elements.resultsKicker.textContent = "Explore your library";
    elements.resultsTitle.textContent = "What do you want to find?";
    elements.resultMeta.textContent = "";
    showSearchState("welcome");
  }

  function setSessionControls(enabled) {
    const controls = [
      elements.searchInput,
      elements.searchSubmit,
      elements.libraryButton,
      elements.folderPaths,
      elements.indexButton,
      elements.welcomeLibraryButton,
      elements.openMediaButton,
      elements.revealMediaButton,
      ...elements.filters,
      ...elements.promptButtons,
    ];
    controls.forEach((control) => {
      if (!control) return;
      if (!enabled || control.dataset.originalHtml === undefined) control.disabled = !enabled;
    });
    elements.searchForm.setAttribute("aria-disabled", String(!enabled));
  }

  function isUnauthorized(error) {
    return error?.status === 401 || error?.status === 403;
  }

  function handleSessionError(error) {
    if (!isUnauthorized(error)) return false;
    renderSessionRequired();
    showToast("This private session is no longer available. Open Lumen from its launcher again.", "error");
    return true;
  }

  async function fetchStatus() {
    clearTimeout(state.statusTimer);
    if (state.statusInFlight) {
      state.statusTimer = window.setTimeout(fetchStatus, 1000);
      return;
    }
    state.statusInFlight = true;
    let nextPoll = document.hidden ? 30000 : 12000;

    try {
      const status = await apiFetch("/api/status");
      const wasRestricted = state.authorized !== true;
      state.authorized = true;
      setSessionControls(true);
      if (wasRestricted) restoreSessionView();
      state.statusFailures = 0;
      renderEngine(status?.engine || {});
      renderLibrary(status?.library || {});
      renderJob(status?.job || null);

      if (status?.job && !terminalJobStates.has(normalizeState(status.job.state))) {
        nextPoll = 1500;
      }
    } catch (error) {
      if (isUnauthorized(error)) {
        renderSessionRequired();
        return;
      }
      state.statusFailures += 1;
      renderEngineError();
      renderConnectionError();
      nextPoll = Math.min(30000, 6000 * state.statusFailures);
    } finally {
      state.statusInFlight = false;
      if (state.authorized !== false) state.statusTimer = window.setTimeout(fetchStatus, nextPoll);
    }
  }

  function renderEngine(engine) {
    const requested = String(engine.requested_device || "CPU");
    const activeDevice = engine.active_device ? String(engine.active_device) : "";
    const displayDevice = activeDevice || requested;
    const backend = String(engine.backend || "Local engine");
    const available = Array.isArray(engine.available_devices) ? engine.available_devices : [];
    const requestedGpu = /gpu|cuda|directml|metal|mps|vulkan/i.test(requested);
    const activeGpu = /gpu|cuda|directml|metal|mps|vulkan/i.test(activeDevice);
    const gpuAvailable = available.some((device) => /gpu|cuda|directml|metal|mps|vulkan/i.test(device));
    const usingFallback = requestedGpu && ((activeDevice && !activeGpu) || (!activeDevice && available.length && !gpuAvailable));

    elements.deviceChip.classList.remove("is-checking", "is-error", "is-ready", "is-fallback");
    if (usingFallback) {
      elements.deviceChip.classList.add("is-fallback");
      elements.deviceLabel.textContent = "CPU fallback";
    } else if (activeDevice || (requestedGpu && gpuAvailable)) {
      elements.deviceChip.classList.add("is-ready");
      elements.deviceLabel.textContent = activeDevice
        ? activeGpu
          ? `${activeDevice} accelerated`
          : `${activeDevice} active`
        : "GPU ready";
    } else {
      elements.deviceChip.classList.add("is-checking");
      elements.deviceLabel.textContent = `${requested} selected`;
    }
    elements.deviceChip.title = [
      `${backend} · ${activeDevice ? `${activeDevice} active` : `${requested} requested`}`,
      available.length ? `Available: ${available.join(", ")}` : "",
      engine.detail || "",
    ]
      .filter(Boolean)
      .join("\n");
    elements.footerEngine.textContent = `${backend} · ${usingFallback ? "CPU fallback" : displayDevice}`;
  }

  function renderEngineError() {
    elements.deviceChip.classList.remove("is-checking", "is-ready", "is-fallback");
    elements.deviceChip.classList.add("is-error");
    elements.deviceLabel.textContent = "Engine offline";
    elements.deviceChip.title = "Lumen could not reach its local engine";
    elements.footerEngine.textContent = "Local engine unavailable";
  }

  function renderLibrary(library) {
    const assets = numeric(library.assets);
    const segments = numeric(library.segments);
    const photos = numeric(library.photos);
    const videos = numeric(library.videos);
    const errors = numeric(library.errors);
    const offline = numeric(library.offline_skipped);

    elements.libraryCount.textContent = compactCount(assets);
    elements.libraryCount.title = `${formatCount(assets)} indexed ${plural(assets, "file")}`;
    elements.segmentCount.textContent = formatCount(segments);
    elements.photoCount.textContent = formatCount(photos);
    elements.videoCount.textContent = formatCount(videos);

    const summary = [`${formatCount(assets)} ${plural(assets, "file")}`];
    if (offline) summary.push(`${formatCount(offline)} offline skipped`);
    if (errors) summary.push(`${formatCount(errors)} ${plural(errors, "error")}`);
    elements.assetSummary.textContent = summary.join(" · ");
  }

  function renderJob(job) {
    if (!job) {
      if (!state.lastJobId) {
        elements.jobPanel.hidden = true;
      } else if (!terminalJobStates.has(state.lastJobState)) {
        elements.jobPanel.hidden = false;
        elements.jobPanel.classList.remove("is-error", "is-indeterminate");
        elements.jobPanel.classList.add("is-complete");
        elements.jobTitle.textContent = "Library is up to date";
        elements.jobProgressLabel.textContent = "100%";
        elements.jobProgressBar.style.width = "100%";
        elements.jobProgressTrack.setAttribute("aria-valuenow", "100");
        elements.jobProgressTrack.setAttribute("aria-valuetext", "Indexing complete");
        elements.jobCurrent.textContent = "Indexing finished";
        elements.jobCounts.textContent = "";
        state.lastJobState = "complete";
        showToast("Library indexing is complete.");
      }
      return;
    }

    const jobState = normalizeState(job.state);
    const total = numeric(job.total);
    const processed = numeric(job.processed);
    const indexed = numeric(job.indexed);
    const skipped = numeric(job.skipped);
    const errors = numeric(job.errors);
    const terminal = terminalJobStates.has(jobState);
    const failed = ["failed", "error", "cancelled", "canceled"].includes(jobState);
    const percent = total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : terminal ? 100 : 0;

    elements.jobPanel.hidden = false;
    elements.jobPanel.classList.toggle("is-complete", terminal && !failed);
    elements.jobPanel.classList.toggle("is-error", failed);
    elements.jobPanel.classList.toggle("is-indeterminate", !terminal && total <= 0);
    elements.jobTitle.textContent = failed
      ? jobState === "cancelled" || jobState === "canceled"
        ? "Indexing was cancelled"
        : "Indexing needs attention"
      : terminal
        ? "Library is up to date"
        : jobState === "queued"
          ? "Indexing queued"
          : "Indexing your library";
    elements.jobProgressLabel.textContent = total > 0 || terminal ? `${percent}%` : "Starting";
    elements.jobProgressBar.style.width = total > 0 || terminal ? `${percent}%` : "8%";
    elements.jobProgressTrack.setAttribute("aria-valuenow", String(percent));
    elements.jobProgressTrack.setAttribute(
      "aria-valuetext",
      total > 0 ? `${processed} of ${total} files processed` : elements.jobProgressLabel.textContent,
    );
    elements.jobCurrent.textContent = terminal
      ? failed
        ? job.current || `${errors} ${plural(errors, "error")} reported`
        : `${formatCount(indexed)} searchable ${plural(indexed, "item")} added`
      : job.current || (jobState === "queued" ? "Waiting for the engine…" : "Scanning folders…");
    elements.jobCounts.textContent = total
      ? `${formatCount(processed)} / ${formatCount(total)} · ${formatCount(skipped)} skipped${errors ? ` · ${formatCount(errors)} errors` : ""}`
      : `${formatCount(indexed)} indexed${skipped ? ` · ${formatCount(skipped)} skipped` : ""}`;

    const sameJob = state.lastJobId === job.id;
    const becameTerminal = terminal && (!sameJob || !terminalJobStates.has(state.lastJobState));
    if (becameTerminal) {
      showToast(
        failed
          ? "Indexing finished with a problem. Check the folder paths and try again."
          : `Library ready—${formatCount(indexed)} ${plural(indexed, "item")} indexed.`,
        failed ? "error" : "success",
      );
    }

    state.lastJobId = job.id;
    state.lastJobState = jobState;
  }

  async function indexFolders(event) {
    event.preventDefault();
    const paths = [
      ...new Set(
        elements.folderPaths.value
          .split(/\r?\n/)
          .map((path) => path.trim().replace(/^(["'])(.*)\1$/, "$2"))
          .filter(Boolean),
      ),
    ];

    if (!paths.length) {
      elements.folderPaths.focus();
      showToast("Add at least one folder path to start indexing.", "error");
      return;
    }

    setButtonBusy(elements.indexButton, true, "Starting…");
    try {
      const payload = await apiFetch("/api/index", {
        method: "POST",
        body: { paths },
      });
      const job = payload?.job || payload;
      if (job) renderJob(job);
      showToast(`${pluralizedFolder(paths.length)} queued for private, local indexing.`);
      clearTimeout(state.statusTimer);
      state.statusTimer = window.setTimeout(fetchStatus, 350);
    } catch (error) {
      if (!handleSessionError(error)) {
        showToast(error.message || "Lumen could not start indexing those folders.", "error");
      }
    } finally {
      setButtonBusy(elements.indexButton, false);
      if (state.authorized !== true) setSessionControls(false);
    }
  }

  async function search(rawQuery) {
    const query = String(rawQuery || "").trim();
    if (!query) {
      elements.searchInput.focus();
      showToast("Describe a moment you want to find.", "error");
      return;
    }

    if (state.searchController) state.searchController.abort();
    const controller = new AbortController();
    state.searchController = controller;
    state.lastQuery = query;
    elements.searchInput.value = query;

    showSearchState("loading");
    elements.resultsKicker.textContent = "Searching your memory";
    elements.resultsTitle.textContent = `Results for “${query}”`;
    elements.resultMeta.textContent = "Looking locally…";

    try {
      const payload = await apiFetch("/api/search", {
        method: "POST",
        body: { q: query, kind: state.kind, limit: 60 },
        signal: controller.signal,
      });
      if (controller !== state.searchController) return;

      const results = Array.isArray(payload?.results) ? payload.results : [];
      renderResults(results);
      elements.resultsKicker.textContent = results.length ? "Closest moments" : "No close matches";
      elements.resultsTitle.textContent = results.length ? `Results for “${query}”` : "That moment is still hiding";
      elements.resultMeta.textContent = `${formatCount(results.length)} ${plural(results.length, "result")}${
        hasNumber(payload?.elapsed_ms) ? ` · ${formatElapsed(payload.elapsed_ms)}` : ""
      }`;

      if (!results.length) {
        elements.emptyMessage.textContent =
          state.kind === "all"
            ? "Try fewer details, a related phrase, or add another folder to your library."
            : `Try searching all moments instead of only ${state.kind === "image" ? "photos" : "videos"}.`;
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      if (handleSessionError(error)) return;
      showSearchState("error");
      elements.resultsKicker.textContent = "Local search unavailable";
      elements.resultsTitle.textContent = "We couldn’t finish that search";
      elements.resultMeta.textContent = "";
      elements.errorMessage.textContent = error.message || "Lumen couldn’t reach its local search engine.";
    } finally {
      if (controller === state.searchController) state.searchController = null;
    }
  }

  function showSearchState(view) {
    elements.welcomeState.hidden = view !== "welcome";
    elements.loadingGrid.hidden = view !== "loading";
    elements.resultGrid.hidden = view !== "results";
    elements.emptyState.hidden = view !== "empty";
    elements.errorState.hidden = view !== "error";
  }

  function renderResults(results) {
    elements.resultGrid.replaceChildren();
    if (!results.length) {
      showSearchState("empty");
      return;
    }

    const fragment = document.createDocumentFragment();
    results.forEach((result, index) => fragment.append(createResultCard(result, index)));
    elements.resultGrid.append(fragment);
    showSearchState("results");
  }

  function createResultCard(result, index) {
    const kind = isVideo(result) ? "video" : "image";
    const name = result.name || fileName(result.path) || (kind === "video" ? "Untitled video" : "Untitled photo");
    const article = document.createElement("article");
    article.className = "media-card";
    article.style.animationDelay = `${Math.min(index, 12) * 28}ms`;

    const previewButton = document.createElement("button");
    previewButton.className = "media-preview-button";
    previewButton.type = "button";
    previewButton.setAttribute("aria-label", `Preview ${name}`);
    previewButton.addEventListener("click", () => openViewer(result));

    const image = document.createElement("img");
    image.className = "media-image";
    image.alt = "";
    image.loading = "lazy";
    image.decoding = "async";
    const thumbnail = result.thumbnail_url || (kind === "image" ? result.media_url : "");
    if (thumbnail) {
      image.src = thumbnail;
      image.addEventListener("error", () => previewButton.classList.add("has-image-error"), { once: true });
    } else {
      previewButton.classList.add("has-image-error");
    }

    const fallback = document.createElement("span");
    fallback.className = "media-fallback";
    fallback.setAttribute("aria-hidden", "true");
    fallback.innerHTML = kind === "video" ? videoIcon : photoIcon;

    const kindBadge = document.createElement("span");
    kindBadge.className = "kind-badge";
    kindBadge.innerHTML = `${kind === "video" ? videoIcon : photoIcon}<span>${kind === "video" ? "Video" : "Photo"}</span>`;

    previewButton.append(image, fallback, kindBadge);

    if (kind === "video" && hasNumber(result.timestamp_ms)) {
      const timestamp = document.createElement("span");
      timestamp.className = "timestamp-badge";
      timestamp.textContent = formatDuration(result.timestamp_ms, true);
      timestamp.title = "Matching moment in video";
      previewButton.append(timestamp);
    }

    if (hasNumber(result.score)) {
      const similarity = document.createElement("span");
      similarity.className = "card-similarity";
      similarity.textContent = `${formatScore(result.score)} similarity`;
      previewButton.append(similarity);
    }

    const body = document.createElement("div");
    body.className = "card-body";

    const titleRow = document.createElement("div");
    titleRow.className = "card-title-row";
    const title = document.createElement("h3");
    title.className = "card-title";
    title.textContent = name;
    title.title = name;

    const moreButton = document.createElement("button");
    moreButton.className = "card-more";
    moreButton.type = "button";
    moreButton.setAttribute("aria-label", `Actions for ${name}`);
    moreButton.setAttribute("aria-expanded", "false");
    moreButton.innerHTML = `
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <circle cx="5" cy="12" r="1"></circle>
        <circle cx="12" cy="12" r="1"></circle>
        <circle cx="19" cy="12" r="1"></circle>
      </svg>`;

    const menu = document.createElement("div");
    menu.className = "card-menu";
    menu.hidden = true;
    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.textContent = "Open file";
    openButton.addEventListener("click", () => performAssetAction("open", result, openButton));
    const revealButton = document.createElement("button");
    revealButton.type = "button";
    revealButton.textContent = "Show in folder";
    revealButton.addEventListener("click", () => performAssetAction("reveal", result, revealButton));
    menu.append(openButton, revealButton);

    moreButton.addEventListener("click", (event) => {
      event.stopPropagation();
      const willOpen = menu.hidden;
      closeCardMenus();
      menu.hidden = !willOpen;
      moreButton.setAttribute("aria-expanded", String(willOpen));
      if (willOpen) openButton.focus();
    });

    titleRow.append(title, moreButton);
    const subtitle = document.createElement("p");
    subtitle.className = "card-subtitle";
    subtitle.textContent = cardSubtitle(result, kind);
    subtitle.title = result.path || subtitle.textContent;
    body.append(titleRow, subtitle);
    article.append(previewButton, body, menu);
    return article;
  }

  function closeCardMenus() {
    $$(".card-menu:not([hidden])").forEach((menu) => {
      menu.hidden = true;
      const toggle = menu.parentElement.querySelector(".card-more");
      if (toggle) toggle.setAttribute("aria-expanded", "false");
    });
  }

  function openViewer(result) {
    state.currentResult = result;
    const kind = isVideo(result) ? "video" : "image";
    const name = result.name || fileName(result.path) || "Media preview";
    elements.viewerKind.textContent = kind === "video" ? "Video moment" : "Photo";
    elements.viewerTitle.textContent = name;
    elements.viewerPath.textContent = result.path || "Local media";
    elements.viewerPath.title = result.path || "";
    renderViewerMeta(result, kind);
    renderViewerMedia(result, kind);

    if (typeof elements.viewer.showModal === "function") {
      elements.viewer.showModal();
    } else {
      elements.viewer.setAttribute("open", "");
    }
  }

  function renderViewerMedia(result, kind) {
    elements.viewerStage.replaceChildren();
    const source = result.media_url || result.thumbnail_url;
    if (!source) {
      renderViewerPlaceholder("Preview unavailable");
      return;
    }

    if (kind === "video") {
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.playsInline = true;
      if (result.thumbnail_url) video.poster = result.thumbnail_url;
      video.src = source;
      video.addEventListener(
        "loadedmetadata",
        () => {
          const seconds = numeric(result.timestamp_ms) / 1000;
          if (seconds > 0 && Number.isFinite(video.duration)) video.currentTime = Math.min(seconds, video.duration);
        },
        { once: true },
      );
      video.addEventListener("error", () => renderViewerPlaceholder("Video preview unavailable"), { once: true });
      elements.viewerStage.append(video);
    } else {
      const image = document.createElement("img");
      image.src = source;
      image.alt = result.name ? `Preview of ${result.name}` : "Photo preview";
      image.addEventListener("error", () => renderViewerPlaceholder("Photo preview unavailable"), { once: true });
      elements.viewerStage.append(image);
    }
  }

  function renderViewerPlaceholder(message) {
    const placeholder = document.createElement("div");
    placeholder.className = "viewer-placeholder";
    placeholder.innerHTML = `${photoIcon}<span></span>`;
    $("span", placeholder).textContent = message;
    elements.viewerStage.replaceChildren(placeholder);
  }

  function renderViewerMeta(result, kind) {
    const dimensions = result.width && result.height ? `${formatCount(result.width)} × ${formatCount(result.height)}` : "Unknown";
    const entries = [
      ["Similarity", hasNumber(result.score) ? formatScore(result.score) : "—"],
      ["Dimensions", dimensions],
      [kind === "video" ? "Matching moment" : "Type", kind === "video" ? formatDuration(result.timestamp_ms, true) : "Image"],
      [kind === "video" ? "Duration" : "Modified", kind === "video" ? formatDuration(result.duration_ms, true) : formatDate(result.modified_at)],
    ];

    elements.viewerMeta.replaceChildren();
    entries.forEach(([term, description]) => {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = description;
      wrap.append(dt, dd);
      elements.viewerMeta.append(wrap);
    });
  }

  function clearViewer() {
    const video = $("video", elements.viewerStage);
    if (video) video.pause();
    elements.viewerStage.replaceChildren();
    state.currentResult = null;
  }

  async function performAssetAction(action, result = state.currentResult, trigger = null) {
    if (!result || result.asset_id === undefined || result.asset_id === null) {
      showToast("That file is no longer available in the library.", "error");
      return;
    }

    const button = trigger || (action === "open" ? elements.openMediaButton : elements.revealMediaButton);
    setButtonBusy(button, true, action === "open" ? "Opening…" : "Finding…");
    try {
      await apiFetch(`/api/${action}/${encodeURIComponent(result.asset_id)}`, { method: "POST" });
      showToast(action === "open" ? "Opened in your default app." : "Revealed in its folder.");
      closeCardMenus();
    } catch (error) {
      if (!handleSessionError(error)) {
        showToast(error.message || `Lumen could not ${action} that file.`, "error");
      }
    } finally {
      setButtonBusy(button, false);
      if (state.authorized !== true) setSessionControls(false);
    }
  }

  function setButtonBusy(button, busy, busyLabel = "Working…") {
    if (!button) return;
    if (busy) {
      button.dataset.originalHtml = button.innerHTML;
      button.disabled = true;
      button.textContent = busyLabel;
    } else {
      button.disabled = false;
      if (button.dataset.originalHtml !== undefined) {
        button.innerHTML = button.dataset.originalHtml;
        delete button.dataset.originalHtml;
      }
    }
  }

  function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast${type === "error" ? " is-error" : ""}`;
    toast.setAttribute("role", type === "error" ? "alert" : "status");
    toast.textContent = message;
    elements.toastRegion.append(toast);

    window.setTimeout(() => {
      toast.classList.add("is-leaving");
      window.setTimeout(() => toast.remove(), 240);
    }, type === "error" ? 6000 : 4200);
  }

  function isVideo(result) {
    return ["video", "movie"].includes(String(result?.kind || "").toLowerCase());
  }

  function cardSubtitle(result, kind) {
    const folder = parentFolder(result.path);
    if (kind === "video" && hasNumber(result.timestamp_ms)) {
      return `${folder || "Video"} · moment at ${formatDuration(result.timestamp_ms, true)}`;
    }
    return folder || formatDate(result.modified_at) || "Local photo";
  }

  function fileName(path) {
    if (!path) return "";
    return String(path).split(/[\\/]/).filter(Boolean).pop() || "";
  }

  function parentFolder(path) {
    if (!path) return "";
    const parts = String(path).split(/[\\/]/).filter(Boolean);
    return parts.length > 1 ? parts.at(-2) : "";
  }

  function normalizeState(value) {
    return String(value || "running").trim().toLowerCase();
  }

  function numeric(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function hasNumber(value) {
    return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  }

  function plural(count, noun) {
    return Number(count) === 1 ? noun : `${noun}s`;
  }

  function pluralizedFolder(count) {
    return `${formatCount(count)} ${plural(count, "folder")}`;
  }

  function formatCount(value) {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(numeric(value));
  }

  function compactCount(value) {
    const count = numeric(value);
    if (count < 1000) return String(count);
    return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(count);
  }

  function formatScore(value) {
    const score = Number(value);
    if (!Number.isFinite(score)) return "—";
    const percent = Math.abs(score) <= 1 ? score * 100 : score;
    return `${Math.round(Math.max(0, Math.min(100, percent)))}%`;
  }

  function formatDuration(value, showHours = false) {
    let totalSeconds = Math.max(0, Math.floor(numeric(value) / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    totalSeconds -= hours * 3600;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (hours || showHours === "always") {
      return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    }
    return `${minutes}:${String(seconds).padStart(2, "0")}`;
  }

  function formatElapsed(value) {
    const milliseconds = numeric(value);
    return milliseconds >= 1000 ? `${(milliseconds / 1000).toFixed(milliseconds >= 10000 ? 0 : 1)} s` : `${Math.round(milliseconds)} ms`;
  }

  function formatDate(value) {
    if (!value) return "Unknown";
    let normalized = value;
    if (typeof value === "number" && value < 1e12) normalized = value * 1000;
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return "Unknown";
    return new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" }).format(date);
  }

  init();
})();
