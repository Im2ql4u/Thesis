# Depth analysis: N=2, omega=1.0, arch=ctnn_vcycle, cusp=True

- params: 9,842; E=2.999729 Ha (+0.228%); overlap^2=0.999954
- NTK numerical rank: 99
- lazy-vs-rich CKA(first,final): 0.9321065051233682
- hidden eff_rank: {'node_embed': 2.6283533819600473, 'edge_embed': 1.3756946174092952, 'node_down': 1.877317292351876, 'edge_down': 1.2948505492711497, 'f_head': 1.0}
- message decode R^2: local_density=0.00, local_coulomb=0.00, nn_distance=0.00, same_spin_count=1.00

Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.
Data: depth_data.npz + data_*.csv + summary.json.
