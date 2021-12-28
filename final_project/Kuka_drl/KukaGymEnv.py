import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import random
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import random
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class KukaGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 100
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=True,
               isDiscrete=True,
               maxSteps = 500, #so buoc toi da cua robot trong 1 episode
               block_angle = 0):
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._block_angle=block_angle

    self._width = 341
    self._height = 256

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
    self.seed()
    self.reset()
    self.observationDim = len(self.getExtendedObservation())

    observation_high = np.array([largeValObservation] * self.observationDim)
    if (self._isDiscrete):
      self.action_dim=4
      self.action_space = spaces.Discrete(self.action_dim)
    else:
       action_dim = 3
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self): #khoi tao lai moi truong moi sau moi episode
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.6400,0.000000,0.000000,0.0,1.0)
    
    self.xpos = 0.5 -0.2*random.random() + 0.07
    self.ypos = -0.2*random.random()+ 0.2
    self.zpos = random.random()/7
    self.ang=self._block_angle+3.14/2
    orn = p.getQuaternionFromEuler([0,0,self.ang])
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), self.xpos,self.ypos,self.zpos,orn[0],orn[1],orn[2],orn[3])
    p.setGravity(0,0,0)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    for down in range(250):
      getdownAction=[0,0,-0.001]
      self._kuka.applyAction(getdownAction,terminate=False)
      p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self): #tra ve khoang cach tu tay kep toi vat
     ## Hien thi camera robot nhung lam giam toc do render
     #viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
     #projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

     #img_arr = p.getCameraImage(width=self._width,height=self._height,viewMatrix=viewMat,projectionMatrix=projMatrix)
     #rgb=img_arr[2]
     #np_img_arr = np.reshape(rgb, (self._height, self._width, 4))

     self._observation = self._kuka.getObservation()

     gripperState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
     gripperPos = gripperState[0]
     gripperOrn = gripperState[1]
     blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)

     invGripperPos,invGripperOrn = p.invertTransform(gripperPos,gripperOrn)
     gripperMat = p.getMatrixFromQuaternion(gripperOrn)
     dir0 = [gripperMat[0],gripperMat[3],gripperMat[6]]
     dir1 = [gripperMat[1],gripperMat[4],gripperMat[7]]
     dir2 = [gripperMat[2],gripperMat[5],gripperMat[8]]

     gripperEul =  p.getEulerFromQuaternion(gripperOrn)
     
     blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
     projectedBlockPos3D =[blockPosInGripper[0],blockPosInGripper[1],blockPosInGripper[2]]
     blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
     

     #we return the relative x,y,z position and euler angle of block in gripper space
     blockInGripperPosXYZEulZ =[blockPosInGripper[0],blockPosInGripper[1],blockPosInGripper[2]]
     x_distance_state=self.xpos-self._kuka.endEffectorPos[0]
     y_distance_state=self.ypos-self._kuka.endEffectorPos[1]
     z_distance_state=self.zpos-self._kuka.endEffectorPos[2]+0.25
   
     distance_state=[x_distance_state,y_distance_state,z_distance_state]
     self._observation=list(distance_state)
     return self._observation

  def step(self, action):# di chuyen moi step
    if (self._isDiscrete):
      dv = 0.01
      dx = [-dv,dv,0,0][action]
      dy = [0,0,-dv,dv][action]
      dz = [dv,0,0,-dv][action]
      # f = -0.05
      realAction = [dx,dy,dz,0,0.4,0]
      # print(realAction)
    else:
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      dz = action[2] * dv
      da = action[3] * 0.05
      f = 0.3
      realAction = [dx,dy,dz,random.uniform(0.05,0.05,0.05),da,f]
    return self.step2( realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action,terminate=self._termination())
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()
    done = self._termination()

    reward = self._reward()
    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
       


    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
   
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    if (self.terminated or self._envStepCounter>self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.22
    if self.xpos-0.015< self._kuka.endEffectorPos[0] < self.xpos + 0.015:
            if self.ypos - 0.015 < self._kuka.endEffectorPos[1] < self.ypos + 0.015:
       
    
              if self.zpos-0.02 < self._kuka.endEffectorPos[2]-0.25 < self.zpos+0.02:
                  self.terminated = 1
                  fingerAngle = 0.6
  # mo phong gap
                  RotateAction = [0,0,0,fingerAngle,3.14-3.14*0.5-self.ang]
                  self._kuka.applyAction(RotateAction,terminate=True)
                  p.stepSimulation()
                  for i in range (1000):
                    graspAction = [0,0,0.000001,fingerAngle,0]
                    self._kuka.applyAction(graspAction,terminate=True)
                    p.stepSimulation()
                    fingerAngle = fingerAngle-(0.6/1000)
                    if (fingerAngle<0):
                      fingerAngle=0

                  for i in range (10000):
                    graspAction = [0,0,0.001,fingerAngle,0]
                    self._kuka.applyAction(graspAction,terminate=True)
                    p.stepSimulation()
                    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
                    if (blockPos[2] > 0.4):
                      break
                    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
                    actualEndEffectorPos = state[0]
                    if (actualEndEffectorPos[2]>0.8):
                      break


              self._observation = self.getExtendedObservation()
              return True
    return False

  def _reward(self):# diem thuong: +3 khi toi vat, +1 khi gap duoc vat va -2khoang cach den vat
    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
    x_distance=abs(self.xpos-self._kuka.endEffectorPos[0])
    y_distance=abs(self.ypos-self._kuka.endEffectorPos[1])
    z_distance=abs(self.zpos-self._kuka.endEffectorPos[2]+0.25)
    total_distance=np.sqrt(x_distance**2+y_distance**2+z_distance**2)
    reward = -2*(total_distance)
    if total_distance<0.06:
      reward = reward+3
    if ((blockPos[2] >0.4 and blockPos[2]<0.5 )and self._kuka.endEffectorPos[2]>0.4 ):
      reward = reward+1
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step