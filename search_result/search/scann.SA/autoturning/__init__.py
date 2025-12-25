"""
Remove the code in __init__.py。
Because when there are any code in __init__.py, I run the script failed
"""


# import os
# import sys
# import time
#
#
# SCRIPT_START_TIME = time.time()
#
#
# def trace_calls(frame, event, arg):
#     if event == 'call':
#         co = frame.f_code
#         func_name = co.co_name
#         if co.co_filename.startswith('<') or \
#                 'site-packages' in co.co_filename or \
#                 '__main__' in sys.modules and sys.modules['__main__'].__package__ not in co.co_filename:
#             return
#
#         file_name = os.path.basename(co.co_filename)
#
#         class_name = None
#         if 'self' in frame.f_locals:
#             self_obj = frame.f_locals['self']
#             if hasattr(self_obj, '__class__'):
#                 class_name = self_obj.__class__.__name__
#         elif 'cls' in frame.f_locals:
#             cls_obj = frame.f_locals['cls']
#             if isinstance(cls_obj, type):
#                 class_name = cls_obj.__name__
#
#         start_time = time.time()
#         print(f'[{start_time-SCRIPT_START_TIME}][{file_name}:{class_name}.{func_name}] start')
#
#         def trace_returns(frame, event, arg):
#             if event == 'return':
#                 end_time = time.time()
#                 print(f'[{end_time-SCRIPT_START_TIME}][{file_name}:{class_name}.{func_name}] end, run {end_time - start_time} seconds.')
#             return trace_returns
#         return trace_returns
#     return None
#
#
# sys.settrace(trace_calls)
#
#
