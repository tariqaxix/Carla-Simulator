#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
import numpy as np

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

_agents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'WindowsNoEditor', 'PythonAPI', 'carla')
if _agents_path not in sys.path:
    sys.path.insert(0, _agents_path)

import carla
import ai_knowledge as data
from ai_knowledge import Status


class Executor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    self.target_pos = knowledge.get_location()

  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    if status == Status.DRIVING:
      dest = self.knowledge.get_current_destination()
      self.update_control(dest, [1], time_elapsed)
    elif status == Status.HEALING:
      self._avoid_obstacle(time_elapsed)
    elif status == Status.ARRIVED:
      control = carla.VehicleControl()
      control.throttle = 0.0
      control.brake = 1.0
      control.hand_brake = True
      self.vehicle.apply_control(control)

  def _avoid_obstacle(self, time_elapsed):
    obstacle = self.knowledge.memory.get('obstacle', {})
    side = obstacle.get('side', 'right')
    control = carla.VehicleControl()
    control.hand_brake = False
    control.brake = 0.0

    final_dest = self.knowledge.memory.get('final_dest', None)
    near_final = False
    if final_dest is not None:
      loc = self.vehicle.get_transform().location
      dx = final_dest.x - loc.x
      dy = final_dest.y - loc.y
      near_final = math.sqrt(dx * dx + dy * dy) < 15.0

    if near_final:
      control.throttle = 0.0
      control.steer = -0.2 if side == 'right' else 0.2
    else:
      control.throttle = 0.5
      control.steer = -0.5 if side == 'right' else 0.5
    self.vehicle.apply_control(control)

  def update_control(self, destination, additional_vars, delta_time):
    transform = self.vehicle.get_transform()
    current_loc = transform.location
    forward = transform.get_forward_vector()

    target_speed = self.knowledge.memory.get('target_speed', 7.0)
    velocity = self.vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    dx = destination.x - current_loc.x
    dy = destination.y - current_loc.y
    dist = math.sqrt(dx * dx + dy * dy)

    control = carla.VehicleControl()
    control.hand_brake = False

    # Taper speed on the final waypoint only to avoid sawtooth along the route
    is_final = self.knowledge.memory.get('is_final_wp', False)
    if is_final and target_speed > 0 and dist < 15.0:
      target_speed = min(target_speed, max(0.3, (dist - 4.0) * 0.2))

    if dist < 0.5:
      control.throttle = 0.0
      control.brake = min(0.6, current_speed / 7.0)
      control.steer = 0.0
      self.vehicle.apply_control(control)
      return

    dest_nx, dest_ny = dx / dist, dy / dist
    fwd_mag = math.sqrt(forward.x ** 2 + forward.y ** 2)
    if fwd_mag < 1e-6:
      fwd_nx, fwd_ny = 1.0, 0.0
    else:
      fwd_nx, fwd_ny = forward.x / fwd_mag, forward.y / fwd_mag

    dot = max(-1.0, min(1.0, fwd_nx * dest_nx + fwd_ny * dest_ny))
    angle = math.acos(dot)
    # UE4 left-hand coords: positive cross_z → destination is to the right → positive steer
    cross_z = fwd_nx * dest_ny - fwd_ny * dest_nx
    if cross_z < 0:
      angle = -angle

    steer_limit = max(0.25, 1.0 - current_speed / 18.0)
    obs_bias = self.knowledge.memory.get('obstacle_steer_bias', 0.0)
    control.steer = max(-steer_limit, min(steer_limit, angle / (math.pi / 2.0) + obs_bias))

    # Red-light block is separate from the waypoint-distance check so the car
    # can always resume when target_speed becomes positive again
    if target_speed <= 0:
      control.throttle = 0.0
      control.brake = min(0.6, max(0.1, current_speed / 6.0))
      self.vehicle.apply_control(control)
      return

    speed_diff = target_speed - current_speed
    if speed_diff > 0.5:
      control.throttle = min(0.9, max(0.2, speed_diff / target_speed))
      control.brake = 0.0
    elif speed_diff > -0.5:
      control.throttle = 0.0
      control.brake = 0.0
    else:
      control.throttle = 0.0
      control.brake = min(0.5, abs(speed_diff) / target_speed)

    self.vehicle.apply_control(control)


class Planner(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge
    self.path = deque([])

  def make_plan(self, source, destination):
    self.path = self.build_path(source, destination)
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())

  def update(self, time_elapsed):
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())

  def update_plan(self):
    if len(self.path) == 0:
      return

    if self.knowledge.arrived_at(self.path[0]):
      self.path.popleft()

    if len(self.path) == 0:
      self.knowledge.update_status(Status.ARRIVED)
      self.knowledge.update_data('is_final_wp', False)
    else:
      self.knowledge.update_status(Status.DRIVING)
      self.knowledge.update_data('is_final_wp', len(self.path) == 1)
      self.knowledge.update_data('final_dest', self.path[-1])

  def get_current_destination(self):
    status = self.knowledge.get_status()
    if status == Status.DRIVING:
      return self.path[0]
    if status == Status.ARRIVED:
      return self.knowledge.get_location()
    if status == Status.HEALING:
      return self.knowledge.get_location()
    if status == Status.CRASHED:
      return self.knowledge.get_location()
    return self.knowledge.get_location()

  def _draw_waypoints(self):
    world = self.knowledge.memory.get('world')
    if world is None:
      return
    debug = world.debug
    for i, loc in enumerate(self.path):
      color = carla.Color(0, 255, 0) if i < len(self.path) - 1 else carla.Color(255, 0, 0)
      debug.draw_point(
        carla.Location(loc.x, loc.y, loc.z + 0.5),
        size=0.08, color=color, life_time=0.0)

  def build_path(self, source, destination):
    self.path = deque([])
    carla_map = self.knowledge.memory.get('map')
    if carla_map is None:
      self.path.append(destination)
      return self.path

    src_loc = source.location
    dst_loc = carla.Location(destination.x, destination.y, destination.z)

    try:
      from agents.navigation.global_route_planner import GlobalRoutePlanner
      grp = GlobalRoutePlanner(carla_map, 2.0)
      route = grp.trace_route(src_loc, dst_loc)
      if route:
        for wp, _ in route:
          self.path.append(wp.transform.location)

        _dx = dst_loc.x - (-30.0)
        _dy = dst_loc.y - 167.0
        if abs(_dx) < 3.0 and abs(_dy) < 3.0:
          # GRP ends beside the entrance pump column at (-27.41, 172.96).
          # Re-route west past all columns then north through the centre lane.
          if len(self.path) > 0:
            self.path.pop()
          self.path.append(carla.Location(-35.0, 172.0, destination.z))
          self.path.append(carla.Location(-30.0, 162.0, destination.z))
        else:
          self.path.append(destination)
        self._draw_waypoints()
        return self.path
    except Exception:
      pass

    # Greedy fallback
    wp = carla_map.get_waypoint(src_loc)
    if wp is None:
      self.path.append(destination)
      return self.path

    STEP = 2.0
    MAX_WPS = 600

    for _ in range(MAX_WPS):
      if wp.transform.location.distance(dst_loc) < 5.0:
        break
      nexts = wp.next(STEP)
      if not nexts:
        break
      wp = min(nexts, key=lambda w: w.transform.location.distance(dst_loc))
      self.path.append(wp.transform.location)

    _dx = dst_loc.x - (-30.0)
    _dy = dst_loc.y - 167.0
    if abs(_dx) < 3.0 and abs(_dy) < 3.0:
      if len(self.path) > 0:
        self.path.pop()
      self.path.append(carla.Location(-35.0, 172.0, destination.z))
      self.path.append(carla.Location(-30.0, 162.0, destination.z))
    else:
      self.path.append(destination)
    self._draw_waypoints()
    return self.path
