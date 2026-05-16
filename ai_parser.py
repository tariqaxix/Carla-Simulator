#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import math
import weakref
import numpy as np
import carla
import ai_knowledge as data


class Monitor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    weak_self = weakref.ref(self)

    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)

    world = self.vehicle.get_world()
    bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    self.lane_detector = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
    self.lane_detector.listen(lambda event: Monitor._on_invasion(weak_self, event))

    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '15')
    lidar_bp.set_attribute('channels', '1')
    lidar_bp.set_attribute('points_per_second', '2000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('upper_fov', '2')
    lidar_bp.set_attribute('lower_fov', '-2')
    lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    self.lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
    self.lidar.listen(lambda lidar_data: Monitor._on_lidar(weak_self, lidar_data))

    self.knowledge.update_data('map', world.get_map())
    self.knowledge.update_data('world', world)
    self.knowledge.update_data('obstacle', {'detected': False, 'side': None})

  def destroy(self):
    for sensor in (self.lane_detector, self.lidar):
      if sensor is not None:
        sensor.stop()
        sensor.destroy()
    self.lane_detector = None
    self.lidar = None

  def update(self, time_elapsed):
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)
    self.knowledge.update_data('speed_limit', self.vehicle.get_speed_limit())

    if self.vehicle.is_at_traffic_light():
      tl = self.vehicle.get_traffic_light()
      if tl is not None:
        self.knowledge.update_data('at_lights', True)
        self.knowledge.update_data('light_state', tl.get_state())
      else:
        self.knowledge.update_data('at_lights', False)
        self.knowledge.update_data('light_state', None)
    else:
      self.knowledge.update_data('at_lights', False)
      self.knowledge.update_data('light_state', None)

    # Actor-query vehicle detection — more reliable than 1-channel LiDAR for cars
    ego_tf = self.vehicle.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    fwd_len = math.sqrt(fwd.x ** 2 + fwd.y ** 2)
    fnx, fny = (fwd.x / fwd_len, fwd.y / fwd_len) if fwd_len > 1e-6 else (1.0, 0.0)

    min_dist = float('inf')
    best_side = None
    world = self.vehicle.get_world()
    for actor in world.get_actors().filter('vehicle.*'):
      if actor.id == self.vehicle.id:
        continue
      v_loc = actor.get_transform().location
      dx = v_loc.x - ego_loc.x
      dy = v_loc.y - ego_loc.y
      dist = math.sqrt(dx * dx + dy * dy)
      if dist > 15.0 or dist < 0.5:
        continue
      if (dx * fnx + dy * fny) < dist * 0.7:
        continue
      if dist < min_dist:
        min_dist = dist
        lat_signed = dx * (-fny) + dy * fnx
        best_side = 'right' if lat_signed > 0 else 'left'

    if min_dist < 15.0:
      self.knowledge.update_data('vehicle_ahead', {'detected': True, 'dist': min_dist, 'side': best_side})
    else:
      self.knowledge.update_data('vehicle_ahead', {'detected': False, 'dist': float('inf'), 'side': None})

  @staticmethod
  def _on_invasion(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.update_data('lane_invasion', event.crossed_lane_markings)

  @staticmethod
  def _on_lidar(weak_self, data):
    self = weak_self()
    if not self:
      return
    raw = np.frombuffer(data.raw_data, dtype=np.float32)
    if raw.size == 0:
      return
    # 4 floats per point: x=forward, y=right, z=up, intensity (sensor frame)
    if raw.size % 4 == 0:
      points = raw.reshape(-1, 4)
    elif raw.size % 3 == 0:
      points = raw.reshape(-1, 3)
      points = np.hstack([points, np.zeros((len(points), 1), dtype=np.float32)])
    else:
      return

    mask = (
      (points[:, 0] > 1.5) &
      (points[:, 0] < 12.0) &
      (np.abs(points[:, 1]) < 2.0) &
      (points[:, 2] > -2.0)
    )
    hits = points[mask]

    # Threshold drops near destination so gas-station bollards (~3-6 hits) are caught
    final_dest = self.knowledge.memory.get('final_dest', None)
    if final_dest is not None:
      loc = self.vehicle.get_transform().location
      dx = final_dest.x - loc.x
      dy = final_dest.y - loc.y
      near_dest = math.sqrt(dx * dx + dy * dy) < 15.0
    else:
      near_dest = False
    threshold = 5 if near_dest else 15

    if len(hits) > threshold:
      avg_y = float(np.mean(hits[:, 1]))
      min_x = float(np.min(hits[:, 0]))
      side = 'right' if avg_y > 0 else 'left'
      self.knowledge.update_data('obstacle', {'detected': True, 'side': side, 'min_x': min_x})
    else:
      self.knowledge.update_data('obstacle', {'detected': False, 'side': None, 'min_x': 12.0})


class Analyser(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge

  def update(self, time_elapsed):
    at_lights = self.knowledge.memory.get('at_lights', False)
    light_state = self.knowledge.memory.get('light_state', None)

    speed_limit_kmh = self.knowledge.memory.get('speed_limit', 30.0)
    cruise_speed = max(8.0, min(13.0, speed_limit_kmh / 3.6))

    is_final = self.knowledge.memory.get('is_final_wp', False)
    final_dest = self.knowledge.memory.get('final_dest', None)
    dist_to_final = float('inf')
    if final_dest is not None:
      loc = self.knowledge.memory.get('location', carla.Location(0, 0, 0))
      dist_to_final = math.sqrt((final_dest.x - loc.x) ** 2 + (final_dest.y - loc.y) ** 2)

    # Slow to a crawl in the final 20 m so the car stops before the gas-station pumps
    if dist_to_final < 20.0:
      cruise_speed = min(cruise_speed, max(0.3, (dist_to_final - 4.0) * 0.15))

    if at_lights and light_state == carla.TrafficLightState.Red:
      self.knowledge.update_data('target_speed', 0.0)
    elif at_lights and light_state == carla.TrafficLightState.Yellow:
      self.knowledge.update_data('target_speed', min(3.0, cruise_speed))
    else:
      self.knowledge.update_data('target_speed', cruise_speed)

    vehicle_ahead = self.knowledge.memory.get('vehicle_ahead', {'detected': False})
    if vehicle_ahead.get('detected') and not (is_final and dist_to_final < 15.0):
      ahead_dist = vehicle_ahead.get('dist', 15.0)
      if ahead_dist < 10.0:
        follow_speed = max(0.0, (ahead_dist - 3.0) * 0.7)
        current_target = self.knowledge.memory.get('target_speed', cruise_speed)
        self.knowledge.update_data('target_speed', min(current_target, follow_speed))

    obstacle = self.knowledge.memory.get('obstacle', {'detected': False})
    if obstacle.get('detected'):
      side = obstacle.get('side', 'right')
      raw_bias = -0.35 if side == 'right' else 0.35
      if is_final and dist_to_final < 5.0:
        bias = 0.0
      elif is_final and dist_to_final < 15.0:
        bias = raw_bias * 0.35
      else:
        bias = raw_bias
      self.knowledge.update_data('obstacle_steer_bias', bias)

      if not (is_final and dist_to_final < 15.0):
        obs_min_x = obstacle.get('min_x', 12.0)
        if obs_min_x < 8.0:
          follow_speed = max(0.0, (obs_min_x - 3.0) * 0.8)
          current_target = self.knowledge.memory.get('target_speed', cruise_speed)
          self.knowledge.update_data('target_speed', min(current_target, follow_speed))
    else:
      self.knowledge.update_data('obstacle_steer_bias', 0.0)
