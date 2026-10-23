"""Warm every active-search branch without disposable setup history."""

from warpack._block_search import ActiveSearch


class ZeroHistorySearch(ActiveSearch):
    def warm(self):
        self.owner.p.zero_()
        return super().warm()
