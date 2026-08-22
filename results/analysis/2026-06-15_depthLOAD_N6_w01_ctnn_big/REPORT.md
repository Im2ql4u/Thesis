# Depth analysis: N=6, omega=0.1, arch=ctnn_vcycle, cusp=True

- params: 79,813; E=3.552866 Ha (-0.021%)
- NTK numerical rank: 1535
- lazy-vs-rich CKA(first,final): None
- hidden eff_rank: {'node_embed': 2.502879292315253, 'edge_embed': 1.2118761507448617, 'node_down': 2.321867306298515, 'edge_down': 4.04087875248888, 'f_head': 1.0}
- message decode R^2: local_density=0.55, local_coulomb=0.89, nn_distance=0.68, same_spin_count=0.24

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
