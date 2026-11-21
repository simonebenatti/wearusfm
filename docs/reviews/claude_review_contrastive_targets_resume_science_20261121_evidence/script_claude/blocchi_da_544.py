import json, sys
d = json.load(open(sys.argv[1]))
for arm in ('CN', 'CR'):
    p = d['arms'][arm]['resume']['pairs']['continuous__reload_a']
    print(arm, 'da init', p['relative_to_left_displacement_from_initial'], 'da 544', p['relative_to_left_displacement_from_544'])
    for part in ('student', 'teacher'):
        print('  ', part, {k: round(v, 6) for k, v in p['by_block_relative_to_left_displacement_from_fork'][part].items()})
    print('   boundary_mismatches', d['arms'][arm]['resume']['boundary_mismatches'], 'missing_active_support', d['arms'][arm]['resume']['missing_active_support'],
          'input_rng_counter_order_differences', {k: len(v['input_rng_counter_order_differences']) for k, v in d['arms'][arm]['resume']['pairs'].items()})
