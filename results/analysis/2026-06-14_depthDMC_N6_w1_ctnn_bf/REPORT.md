# Depth analysis: N=6, omega=1.0, arch=ctnn_vcycle, cusp=True

- params: 12,373; E=20.159125 Ha (+0.006%)
- NTK numerical rank: 1008
- lazy-vs-rich CKA(first,final): 0.6015619080238278
- hidden eff_rank: {'node_embed': 2.6199608940068146, 'edge_embed': 1.2761521514040295, 'node_down': 2.149419583557831, 'edge_down': 1.5841850314380785, 'f_head': 1.0}
- message decode R^2: local_density=0.71, local_coulomb=0.73, nn_distance=0.82, same_spin_count=0.27

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
