"""Strip a configured public root path from incoming requests when a proxy preserves it."""


class StripRootPathMiddleware:
    def __init__(self, app, root_path: str = ""):
        self.app = app
        prefix = (root_path or "").strip()
        if prefix and not prefix.startswith("/"):
            prefix = f"/{prefix}"
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.prefix:
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        if path.startswith(self.prefix) and (path == self.prefix or path.startswith(f"{self.prefix}/")):
            new_path = path[len(self.prefix):] or "/"
            raw_path = scope.get("raw_path")
            if isinstance(raw_path, bytes) and raw_path.startswith(self.prefix.encode("utf-8")):
                new_raw = raw_path[len(self.prefix):] or b"/"
            else:
                new_raw = new_path.encode("utf-8")
            scope = dict(scope)
            scope["path"] = new_path
            scope["raw_path"] = new_raw
        await self.app(scope, receive, send)