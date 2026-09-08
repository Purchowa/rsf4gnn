import json
import os
from tensorboardX import SummaryWriter


class Logger:
    """
    Record the results and store them to the json file
    """
    def __init__(self, model_, data_, log_dir, log_filename = 'log.json'):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log = {
            'model': model_,
            'data': data_,
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'epoch': [],
            'meta': {},
            'summary': {},
            'diagnostics': {
                'status': 'unavailable',  # 'success', 'partial', or 'unavailable'
                'counters': {},
                'metadata': {}
            }
        }
        self.exp_name = model_ + '_' + data_
        json_dir = os.path.join(log_dir, self.exp_name)
        self.writer = SummaryWriter(log_dir=json_dir)
        if not os.path.exists(json_dir):
            os.mkdir(json_dir)
        self.log_file = os.path.join(json_dir, log_filename)

    def add_scalar(self, key, value, epoch):
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if isinstance(value, (int, float)):
            self.writer.add_scalar(tag=key, scalar_value=value, global_step=epoch)

    def set_meta(self, key, value):
        self.log['meta'][key] = value

    def set_summary(self, key, value):
        self.log['summary'][key] = value

    def set_diagnostics_status(self, status):
        """
        Set the diagnostics capture status.
        
        Args:
            status (str): One of 'success', 'partial', or 'unavailable'
        """
        if status not in ['success', 'partial', 'unavailable']:
            raise ValueError(f"Invalid diagnostics status: {status}. Must be one of: 'success', 'partial', 'unavailable'")
        self.log['diagnostics']['status'] = status

    def set_diagnostics_counter(self, key, value):
        """
        Set a single hardware counter value.
        
        Args:
            key (str): Counter name (e.g., 'L1-dcache-load-misses', 'LLC-loads')
            value: Counter value (int or float)
        """
        self.log['diagnostics']['counters'][key] = value

    def set_diagnostics_counters(self, counters_dict):
        """
        Set multiple hardware counter values at once.
        
        Args:
            counters_dict (dict): Dictionary of counter_name -> value pairs
        """
        if not isinstance(counters_dict, dict):
            raise TypeError("counters_dict must be a dictionary")
        self.log['diagnostics']['counters'].update(counters_dict)

    def set_diagnostics_meta(self, key, value):
        """
        Set a metadata field in the diagnostics block.
        
        Args:
            key (str): Metadata key
            value: Metadata value
        """
        self.log['diagnostics']['metadata'][key] = value

    def write(self):
        with open(self.log_file, 'w') as file:
            json.dump(self.log, file)
        self.writer.close()
        