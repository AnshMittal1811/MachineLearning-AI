def update_log_stats_dict(stats, stats_to_add, epoch, name):
    stats.update({**{f'{name}_{k}': v for k, v in stats_to_add.items()}, 'epoch': epoch})

def log_stats_to_tboard(log_writer, stats, epoch, name):
    if log_writer is not None:
        log_writer.add_scalar(f'perf/{name}_acc1', stats['acc1'], epoch)
        log_writer.add_scalar(f'perf/{name}_acc5', stats['acc5'], epoch)
        log_writer.add_scalar(f'perf/{name}_loss', stats['loss'], epoch)