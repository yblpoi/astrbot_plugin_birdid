"""Bird identification helpers shared by the plugin entry point.

The package is split by concern so each part stays testable without a running
AstrBot instance:

* bird.names -- resolve scientific names to Chinese and English names.
* bird.client -- talk to the local identification service.
* bird.media -- pull image and audio payloads out of chat messages.
* bird.render -- cut the detected box and draw spectrograms.
* bird.archive -- rotate observation media into training archives.
"""

__all__ = ["archive", "client", "media", "names", "render"]
