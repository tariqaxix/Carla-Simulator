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

import carla
from enum import Enum

class Status(Enum):
  ARRIVED = 0
  DRIVING = 1
  CRASHED = 2
  HEALING = 3
  UNDEFINED = 4

class Knowledge(object):
  def __init__(self):
    self.status = Status.ARRIVED
    self.memory = {'location': carla.Vector3D(0.0, 0.0, 0.0)}
    self.destination = self.get_location()
    self.status_changed = lambda *_, **__: None
    self.destination_changed = lambda *_, **__: None
    self.data_changed = lambda *_, **__: None

  def set_data_changed_callback(self, callback):
    self.data_changed = callback

  def set_status_changed_callback(self, callback):
    self.status_changed = callback

  def set_destination_changed_callback(self, callback):
    self.destination_changed = callback

  def get_status(self):
    return self.status

  def set_status(self, new_status):
    self.status = new_status

  def get_current_destination(self):
    return self.destination

  def retrieve_data(self, data_name):
    return self.memory[data_name]

  def update_status(self, new_status):
    if (self.status != Status.CRASHED or new_status == Status.HEALING) and self.status != new_status:
      self.set_status(new_status)
      self.status_changed(new_status)

  def get_location(self):
    return self.retrieve_data('location')

  def arrived_at(self, destination):
    return self.distance(self.get_location(), destination) < 5.0

  def update_destination(self, new_destination):
    if self.distance(self.destination, new_destination) > 5.0:
      self.destination = new_destination
      self.destination_changed(new_destination)

  def update_data(self, data_name, pars):
    self.memory[data_name] = pars
    self.data_changed(data_name)

  def distance(self, vec1, vec2):
    l1 = carla.Location(vec1)
    l2 = carla.Location(vec2)
    return l1.distance(l2)
