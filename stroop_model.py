"""replicate the stroop model from cohen et al (1990)
clr = color, wrd = word
i = input, h = hidden, o = output
tc/tw = mapping from taks input to color naming / word reading
"""
import psyneulink as pnl

# CONSTANTS
N_UNITS = 2


def get_stroop_model():
    # model params
    # TODO bad practice, to be moved
    hidden_func = pnl.Logistic(gain=1.0, x_0=4.0)
    unit_noise_std = .01
    dec_noise_std = .1
    integration_rate = .2
    leak = 0
    competition = 1
    # lca_mvn = [0, 1]
    # input layer, color and word
    inp_clr = pnl.TransferMechanism(
        size=N_UNITS, function=pnl.Linear, name='COLOR INPUT'
    )
    inp_wrd = pnl.TransferMechanism(
        size=N_UNITS, function=pnl.Linear, name='WORD INPUT'
    )
    # task layer, represent the task instruction; color naming / word reading
    inp_task = pnl.TransferMechanism(
        size=N_UNITS, function=pnl.Linear, name='TASK'
    )
    # hidden layer for color and word
    hid_clr = pnl.TransferMechanism(
        size=N_UNITS,
        function=hidden_func,
        integrator_mode=True,
        integration_rate=integration_rate,
        noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
        name='COLORS HIDDEN'
    )
    hid_wrd = pnl.TransferMechanism(
        size=N_UNITS,
        function=hidden_func,
        integrator_mode=True,
        integration_rate=integration_rate,
        noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
        name='WORDS HIDDEN'
    )
    # output layer
    output = pnl.TransferMechanism(
        size=N_UNITS,
        function=pnl.Logistic,
        integrator_mode=True,
        integration_rate=integration_rate,
        noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
        name='OUTPUT'
    )
    # decision layer, some accumulator
    decision = pnl.LCAMechanism(
        size=N_UNITS,
        leak=leak,
        competition=competition,
        # MAX_VS_NEXT=lca_mvn,
        noise=pnl.UniformToNormalDist(
            standard_deviation=dec_noise_std).function,
        name='DECISION'
    )

    # PROJECTIONS, weights copied from cohen et al (1990)
    wts_clr_ih = pnl.MappingProjection(
        matrix=[[2.2, -2.2], [-2.2, 2.2]], name='COLOR INPUT TO HIDDEN')
    wts_wrd_ih = pnl.MappingProjection(
        matrix=[[2.6, -2.6], [-2.6, 2.6]], name='WORD INPUT TO HIDDEN')
    wts_clr_ho = pnl.MappingProjection(
        matrix=[[1.3, -1.3], [-1.3, 1.3]], name='COLOR HIDDEN TO OUTPUT')
    wts_wrd_ho = pnl.MappingProjection(
        matrix=[[2.5, -2.5], [-2.5, 2.5]], name='WORD HIDDEN TO OUTPUT')
    wts_tc = pnl.MappingProjection(
        matrix=[[4.0, 4.0], [0, 0]], name='COLOR NAMING')
    wts_tw = pnl.MappingProjection(
        matrix=[[0, 0], [4.0, 4.0]], name='WORD READING')

    # build the model
    model = pnl.Composition(name='STROOP model')
    model.add_node(inp_clr)
    model.add_node(inp_wrd)
    model.add_node(hid_clr)
    model.add_node(hid_wrd)
    model.add_node(inp_task)
    model.add_node(output)
    model.add_node(decision)
    model.add_linear_processing_pathway([inp_clr, wts_clr_ih, hid_clr])
    model.add_linear_processing_pathway([inp_wrd, wts_wrd_ih, hid_wrd])
    model.add_linear_processing_pathway([hid_clr, wts_clr_ho, output])
    model.add_linear_processing_pathway([hid_wrd, wts_wrd_ho, output])
    model.add_linear_processing_pathway([inp_task, wts_tc, hid_clr])
    model.add_linear_processing_pathway([inp_task, wts_tw, hid_wrd])
    model.add_linear_processing_pathway([output, pnl.IDENTITY_MATRIX, decision])

    # LOGGING
    hid_clr.set_log_conditions('value')
    hid_wrd.set_log_conditions('value')
    output.set_log_conditions('value')
    # collect the node handles
    nodes = [inp_clr, inp_wrd, inp_task, hid_clr, hid_wrd, output, decision]
    metadata = [integration_rate, dec_noise_std, unit_noise_std]
    return model, nodes, metadata
