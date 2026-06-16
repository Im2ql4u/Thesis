# Depth analysis: N=6, omega=0.5, arch=ctnn_vcycle, cusp=True

- params: 79,813; E=11.781337 Ha (-0.033%)
- NTK numerical rank: 1535
- lazy-vs-rich CKA(first,final): None
- hidden eff_rank: {'node_embed': 2.6284120157210644, 'edge_embed': 1.28273739736037, 'node_down': 2.4573202074351976, 'edge_down': 3.4215990324127623, 'f_head': 1.0}
- message decode R^2: local_density=0.66, local_coulomb=0.91, nn_distance=0.85, same_spin_count=0.42

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
