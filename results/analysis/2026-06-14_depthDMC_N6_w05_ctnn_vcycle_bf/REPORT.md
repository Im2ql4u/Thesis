# Depth analysis: N=6, omega=0.5, arch=ctnn_vcycle, cusp=True

- params: 12,373; E=11.790515 Ha (+0.057%)
- NTK numerical rank: 1023
- lazy-vs-rich CKA(first,final): 0.4994277146861944
- hidden eff_rank: {'node_embed': 2.552854748016663, 'edge_embed': 1.2564693601217765, 'node_down': 1.6694792173630464, 'edge_down': 1.7400432770224457, 'f_head': 1.0}
- message decode R^2: local_density=0.71, local_coulomb=0.65, nn_distance=0.80, same_spin_count=0.26

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
