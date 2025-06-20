from .base import BaseDatastore 
from .mdp import MDPDatastore 

DATASTORE_CLASSES = [
    MDPDatastore,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}


def init_datastore(datastore_kind, config_path):
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(config_path=config_path)

    return datastore