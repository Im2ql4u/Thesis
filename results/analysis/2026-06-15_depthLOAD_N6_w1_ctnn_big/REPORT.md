# Depth analysis: N=6, omega=1.0, arch=ctnn_vcycle, cusp=True

- params: 79,813; E=20.160535 Ha (-0.001%)
- NTK numerical rank: 1535
- lazy-vs-rich CKA(first,final): None
- hidden eff_rank: {'node_embed': 2.6606739958716625, 'edge_embed': 1.331843389505787, 'node_down': 2.5365703510503965, 'edge_down': 2.9941032135089323, 'f_head': 1.0}
- message decode R^2: local_density=0.73, local_coulomb=0.82, nn_distance=0.91, same_spin_count=0.53

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
