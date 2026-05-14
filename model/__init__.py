import logging
logger = logging.getLogger('base')


def create_model(opt,local_rank):
    from model.model import Multistage_Reg_Fus_Model
    diff_model = Multistage_Reg_Fus_Model(opt,local_rank)
    logger.info('Model [{:s}] is created.'.format(diff_model.__class__.__name__))
    return diff_model