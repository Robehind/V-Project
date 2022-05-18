from typing import List
from ai2thor.controller import Controller
from .utils import AgentPoseState
from ai2thor.platform import CloudRendering
import cv2
import numpy as np
import os
FPS = 25.
ACT_FRAME = 13
PAUSE_FRAME = 13


class OriThorForVis:
    def __init__(
        self,
        width,
        height,
        grid_size,
        rotate_angle
    ) -> None:
        self.ctrler = Controller(
            width=width, height=height,
            rotateStepDegrees=rotate_angle, snapToGrid=False,
            gridSize=grid_size, platform=CloudRendering)
        self.clear_frames()

    def clear_frames(self):
        self.fp_frames = []
        self.top_frames = []

    def top_view(self, scene):
        self.ctrler.reset(scene=scene)
        event = self.ctrler.step(action='GetMapViewCameraProperties')
        # move agent out of scene
        self.ctrler.step(
            action="Teleport", position=dict(y=-10), forceAction=True)
        camera_params = event.metadata['actionReturn']
        event = self.ctrler.step(
            action='AddThirdPartyCamera', **camera_params)
        pic = event.third_party_camera_frames[0]
        return camera_params, pic[:, :, [2, 1, 0]]

    def Animateframe(
        self,
        pos1: AgentPoseState,
        pos2: AgentPoseState,
        birdView: bool,
        smooth: bool = False,
    ):
        if not smooth:
            evt = self.ctrler.step(
                action="Teleport",
                position=pos2.position(),
                rotation=dict(x=0, y=pos2.rotation, z=0),
                horizon=pos2.horizon,
                forceAction=True)
            self.fp_frames.append(evt.cv2img)
            if birdView:
                pic = evt.third_party_camera_frames[0]
                self.top_frames.append(pic[:, :, [2, 1, 0]])
        elif pos1 is None:
            evt = self.ctrler.step(
                action="Teleport",
                position=pos2.position(),
                rotation=dict(x=0, y=pos2.rotation, z=0),
                horizon=pos2.horizon,
                forceAction=True)
        else:
            y = pos2.y
            x_list = np.linspace(pos1.x, pos2.x, ACT_FRAME)
            z_list = np.linspace(pos1.z, pos2.z, ACT_FRAME)
            # 确保转小角
            pos2r = (pos2.rotation + 360) % 360
            pos1r = (pos1.rotation + 360) % 360
            diff = abs(pos2r-pos1r)
            if diff > 360 - diff:
                if pos1r > pos2r:
                    pos1r -= 360
                else:
                    pos2r -= 360
            r_list = np.linspace(pos1r, pos2r, ACT_FRAME)
            h_list = np.linspace(pos1.horizon, pos2.horizon, ACT_FRAME)
            for i in range(ACT_FRAME):
                evt = self.ctrler.step(
                    action="Teleport",
                    position=dict(x=x_list[i], y=y, z=z_list[i]),
                    rotation=dict(y=round(r_list[i])),
                    horizon=round(h_list[i]),
                    forceAction=True)
                self.fp_frames.append(evt.cv2img)
                if birdView:
                    pic = evt.third_party_camera_frames[0]
                    self.top_frames.append(pic[:, :, [2, 1, 0]])

    def get_frames(
        self,
        scene: str,
        poses: List[str],
        birdView=False,
        smooth=False
    ):
        evt = self.ctrler.reset(scene=scene)
        Y = evt.metadata['agent']['position']['y']
        self.ctrler.step("PausePhysicsAutoSim")
        if birdView:
            evt = self.ctrler.step(action='GetMapViewCameraProperties')
            self.ctrler.step(
                action='AddThirdPartyCamera', **evt.metadata['actionReturn'])
        last_p = None
        for p in poses:
            agt_p = AgentPoseState(pose_str=p, y=Y)
            self.Animateframe(last_p, agt_p, birdView, smooth)
            last_p = agt_p

    def visualize(
        self,
        scene: str,
        poses: List[str],
        wait: int = 0,
        birdView: bool = False,
        smooth: bool = False
    ):
        assert wait >= 0
        self.get_frames(scene, poses, birdView, smooth)
        print("Press key to start")
        print("Start pose:", poses[0])
        cv2.imshow('vis', self.fp_frames[0])
        if birdView:
            cv2.imshow("Topview", self.top_frames[0])
        cv2.waitKey(0)
        for i, p in enumerate(poses[1:]):
            if not smooth:
                cv2.imshow('vis', self.fp_frames[i])
                if birdView:
                    cv2.imshow("Topview", self.top_frames[i])
                print("Pose:", p)
                cv2.waitKey(int(wait*1000))
            else:
                for j in range(i*ACT_FRAME, (i+1)*ACT_FRAME):
                    cv2.imshow('vis', self.fp_frames[j])
                    if birdView:
                        cv2.imshow("Topview", self.top_frames[j])
                    cv2.waitKey(int(1000/FPS))
                print("Pose:", p)
                cv2.waitKey(int(wait*1000))
        print("finished")
        cv2.destroyAllWindows()

    def export_video(self, path: str, smooth: bool, prefix: str):
        for ff in ['fp_frames', 'top_frames']:
            frames = getattr(self, ff)
            if frames != []:
                _path = os.path.join(path, prefix+ff+'.mp4')
                if os.path.exists(_path):
                    os.remove(_path)
                video = cv2.VideoWriter(
                    _path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS,
                    (frames[0].shape[1], frames[0].shape[0]))
                for i in range(len(frames)):
                    video.write(frames[i])
                    if (not smooth) or (i % ACT_FRAME == 0 and smooth):
                        # 一个动作完成后等待那么一会儿
                        for j in range(PAUSE_FRAME):
                            video.write(frames[i])
                cv2.destroyAllWindows()
                video.release()
        print("Save Success")
