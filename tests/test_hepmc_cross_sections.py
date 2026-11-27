"""Per-member HepMC rates must describe the exported subset, not its parent run."""
import json

import numpy as np
import pytest

hepmc = pytest.importorskip('pyhepmc')
from maxent_upgrade.hepmc import attach_weights


def _record(number, cross_sections=True):
    event = hepmc.GenEvent()
    event.event_number = number
    event.run_info = hepmc.GenRunInfo()
    event.run_info.weight_names = ['original', 'original_scale']
    event.weights = [float(number), 1.5*number]
    vertex = hepmc.GenVertex()
    vertex.add_particle_in(hepmc.GenParticle((0, 0, 100, 100), 21, 4))
    vertex.add_particle_in(hepmc.GenParticle((0, 0, -100, 100), 21, 4))
    vertex.add_particle_out(hepmc.GenParticle((0, 0, 0, 200), 25, 1))
    event.add_vertex(vertex)
    if cross_sections:
        event.cross_section = hepmc.GenCrossSection()
        event.cross_section.set_cross_section(100., 2., 3, 4)
        event.cross_section.set_xsec(1, 125.)
        event.cross_section.set_xsec_err(1, 3.)
        if cross_sections == 'scalar':
            # Historical files may carry one rate despite multiple weights.
            assert event.cross_section.from_string('100 2 3 4')
    return event


@pytest.mark.parametrize('cross_sections', [True, False, 'scalar'])
def test_member_cross_sections_preserve_originals_and_match_filtered_sums(tmp_path, cross_sections):
    source, output = tmp_path/'source.hepmc', tmp_path/'subset.hepmc'
    with hepmc.open(source, 'w') as writer:
        for n in (1, 2, 3):
            writer.write(_record(n, cross_sections))
    attach_weights(source, output, np.array([3, 1]), np.array([.9, .4]),
                   variations_pb={'scale_up':np.array([1.8, .2])})
    with hepmc.open(output) as reader:
        events = list(reader)
    assert [e.event_number for e in events] == [1, 3]
    for e in events:
        assert e.weights[:2] == [float(e.event_number), 1.5*e.event_number]
        assert e.cross_section.xsec(0) == (100. if cross_sections else -1.)
        second_rate = 100. if cross_sections == 'scalar' else (125. if cross_sections else -1.)
        assert e.cross_section.xsec(1) == second_rate
        assert e.cross_section.xsec(2) == pytest.approx(1.3)
        assert e.cross_section.xsec(3) == pytest.approx(2.)
        assert e.cross_section.xsec_err(2) == 0.
        if cross_sections:
            assert e.cross_section.xsec_err(0) == 2.
            assert e.cross_section.xsec_err(1) == (2. if cross_sections == 'scalar' else 3.)
            assert e.cross_section.accepted_events == 3
            assert e.cross_section.attempted_events == 4
            assert 'maxent_original_GenCrossSection' in e.attributes
        metadata = json.loads(str(e.run_info.attributes['maxent_upgrade']))
        assert metadata['selected_records'] == 2
    for i in (2, 3):
        assert sum(e.weights[i] for e in events) == pytest.approx(events[0].cross_section.xsec(i))
