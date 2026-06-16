# Depth analysis: N=6, omega=0.01, arch=ctnn_vcycle, cusp=True

- params: 79,813; E=0.689368 Ha (-0.107%)
- NTK numerical rank: 1535
- lazy-vs-rich CKA(first,final): None
- hidden eff_rank: {'node_embed': 2.3427058810639414, 'edge_embed': 1.1655851401336341, 'node_down': 2.6812995538533118, 'edge_down': 3.295808265070167, 'f_head': 1.0}
- message decode R^2: local_density=0.13, local_coulomb=0.44, nn_distance=0.17, same_spin_count=0.05

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
