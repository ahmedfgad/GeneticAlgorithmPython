// Show recent release notes first, with older notes available on the same page.
// All notes remain in the HTML for documentation search, printing, and readers
// without JavaScript. Existing release links reveal their target automatically.
window.addEventListener("DOMContentLoaded", function () {
  var releaseHistory = document.getElementById("release-history");
  if (!releaseHistory) {
    return;
  }

  var releaseSections = Array.from(releaseHistory.children).filter(function (element) {
    return element.tagName === "SECTION";
  });
  var releasesPerPage = 10;
  if (releaseSections.length <= releasesPerPage) {
    return;
  }

  var visibleReleaseCount = releasesPerPage;
  var contentsLinks = Array.from(document.querySelectorAll(".toc-tree a"));
  var navigation = document.createElement("div");
  navigation.className = "release-history-navigation";
  var jumpLabel = document.createElement("label");
  jumpLabel.htmlFor = "release-history-version";
  jumpLabel.textContent = "Jump to a release";
  navigation.appendChild(jumpLabel);
  var releaseSelector = document.createElement("select");
  releaseSelector.id = "release-history-version";
  var placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a release";
  releaseSelector.appendChild(placeholder);
  releaseSections.forEach(function (section) {
    var option = document.createElement("option");
    option.value = section.id;
    option.textContent = section.querySelector("h2").firstChild.textContent.trim();
    releaseSelector.appendChild(option);
  });
  navigation.appendChild(releaseSelector);
  releaseHistory.insertBefore(navigation, releaseSections[0]);

  var controls = document.createElement("div");
  controls.className = "release-history-controls";
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", "Older release notes");

  var status = document.createElement("p");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  controls.appendChild(status);

  var showMoreButton = document.createElement("button");
  showMoreButton.type = "button";
  showMoreButton.textContent = "Show 10 more releases";
  controls.appendChild(showMoreButton);

  var showAllButton = document.createElement("button");
  showAllButton.type = "button";
  showAllButton.textContent = "Show all releases";
  controls.appendChild(showAllButton);
  releaseHistory.appendChild(controls);

  function updateVisibleReleases() {
    releaseSections.forEach(function (section, index) {
      section.hidden = index >= visibleReleaseCount;
    });
    contentsLinks.forEach(function (link) {
      var section = document.getElementById(link.hash.slice(1));
      var item = link.closest("li");
      if (section && releaseSections.indexOf(section) !== -1 && item) {
        item.hidden = section.hidden;
      }
    });
    var remainingReleaseCount = releaseSections.length - visibleReleaseCount;
    status.textContent = "Showing " + visibleReleaseCount + " of " +
      releaseSections.length + " release entries.";
    var nextReleaseCount = Math.min(releasesPerPage, remainingReleaseCount);
    showMoreButton.textContent = "Show " + nextReleaseCount + " more " +
      (nextReleaseCount === 1 ? "release" : "releases");
    showMoreButton.hidden = remainingReleaseCount === 0;
    showAllButton.hidden = remainingReleaseCount === 0;
  }

  function showOlderReleases(count) {
    var firstNewSection = releaseSections[visibleReleaseCount];
    visibleReleaseCount = Math.min(count, releaseSections.length);
    updateVisibleReleases();
    // Move keyboard focus to the newly revealed notes, rather than leaving
    // it on a button that moved below them or disappeared after "Show all".
    var heading = firstNewSection.querySelector("h2");
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
    heading.scrollIntoView({ block: "start" });
  }

  function revealLinkedRelease() {
    var targetId;
    try {
      targetId = decodeURIComponent(window.location.hash.slice(1));
    } catch (error) {
      return;
    }
    var target = document.getElementById(targetId);
    var releaseIndex = releaseSections.findIndex(function (section) {
      return target && section.contains(target);
    });
    if (releaseIndex === -1) {
      return;
    }
    releaseSelector.value = releaseSections[releaseIndex].id;
    if (releaseIndex >= visibleReleaseCount) {
      visibleReleaseCount = Math.min(releaseSections.length,
        Math.ceil((releaseIndex + 1) / releasesPerPage) * releasesPerPage);
      updateVisibleReleases();
    }
    target.scrollIntoView({ block: "start" });
  }

  showMoreButton.addEventListener("click", function () {
    showOlderReleases(visibleReleaseCount + releasesPerPage);
  });
  showAllButton.addEventListener("click", function () {
    showOlderReleases(releaseSections.length);
  });
  releaseSelector.addEventListener("change", function () {
    if (releaseSelector.value) {
      var releaseHash = "#" + releaseSelector.value;
      window.location.hash = releaseHash;
      revealLinkedRelease();
    }
  });
  window.addEventListener("hashchange", revealLinkedRelease);
  updateVisibleReleases();
  revealLinkedRelease();
});
