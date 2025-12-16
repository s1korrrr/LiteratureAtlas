import unittest

import numpy as np
import pandas as pd

from analytics.rebuild_analytics import cluster_stability_multi_seed_kmeans, paper_layout_quality_metrics


class RebuildAnalyticsUnitTests(unittest.TestCase):
    def test_paper_layout_quality_small_n_does_not_crash(self):
        n = 9
        df = pd.DataFrame({"paper_id": [f"p{i}" for i in range(n)]})
        rng = np.random.default_rng(seed=0)
        Z = rng.standard_normal((n, 8), dtype=np.float32)

        summary, distort = paper_layout_quality_metrics(df, Z, k=15, max_n=4000, seed=0)

        self.assertTrue(summary.get("available"), summary)
        self.assertLessEqual(summary.get("k", 999), n - 2)
        self.assertEqual(len(distort), n)

    def test_cluster_stability_multi_seed_runs(self):
        # 2 well-separated clusters, 2 points each.
        df = pd.DataFrame(
            {
                "paper_id": ["a", "b", "c", "d"],
                "cluster_id": [0, 0, 1, 1],
            }
        )
        Z = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        stability = cluster_stability_multi_seed_kmeans(df, Z, runs=5, seed=0)

        self.assertTrue(stability.get("available"), stability)
        per_paper = stability.get("per_paper") or []
        self.assertEqual(len(per_paper), 4)
        # Confidence should be non-trivial for obvious clusters.
        self.assertGreaterEqual(float(per_paper[0].get("cluster_confidence", 0.0)), 0.6)


if __name__ == "__main__":
    unittest.main()

