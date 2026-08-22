# Depth analysis: N=6, omega=1.0, arch=ctnn_vcycle, cusp=True

- params: 9,842; E=20.178530 Ha (+0.089%)
- NTK numerical rank: 585
- lazy-vs-rich CKA(first,final): 0.9355083030761503
- hidden eff_rank: {'node_embed': 2.728896789045426, 'edge_embed': 1.2621699530839932, 'node_down': 2.490388779350273, 'edge_down': 1.885554390406095, 'f_head': 1.0}
- message decode R^2: local_density=0.72, local_coulomb=0.54, nn_distance=0.88, same_spin_count=0.31

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
