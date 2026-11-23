"""Bird identification helpers shared by the plugin entry point.

The package is split by concern so each part stays testable without a running
AstrBot instance:

* bird.names -- resolve scientific names to Chinese and English names.
* bird.client -- talk to the local identification service.
* bird.media -- pull image and audio payloads out of chat messages.
* bird.render -- cut the detected box and draw spectrograms.
* bird.archive -- rotate observation media into training archives.
* bird.settings -- normalise the runtime settings and the access policy.
* bird.stats -- bounded counters backing the dashboard.
* bird.hostinfo -- read process and host memory figures.
* bird.dashboard -- assemble the documents the plugin page renders.
"""

__all__ = [
    "archive",
    "client",
    "dashboard",
    "hostinfo",
    "media",
    "names",
    "render",
    "settings",
    "stats",
]
