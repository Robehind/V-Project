from typing import List
from ai2thor.controller import Controller
from .agent_pose_state import AgentPoseState
from ai2thor.platform import CloudRendering
import cv2
import numpy as np
import os
FPS = 25.
ACT_FRAME = 13
PAUSE_FRAME = 13
SAFE_Y = 0.91


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
                horizon=pos2.horizon
            )
            assert evt.metadata['lastActionSuccess'],\
                evt.metadata['errorMessage']
            self.fp_frames.append(evt.cv2img)
            if birdView:
                pic = evt.third_party_camera_frames[0]
                self.top_frames.append(pic[:, :, [2, 1, 0]])
        elif pos1 is None:
            evt = self.ctrler.step(
                action="Teleport",
                position=pos2.position(),
                rotation=dict(x=0, y=pos2.rotation, z=0),
                horizon=pos2.horizon
            )
            assert evt.metadata['lastActionSuccess'],\
                evt.metadata['errorMessage']
        else:
            x_list = np.linspace(pos1.x, pos2.x, ACT_FRAME)
            z_list = np.linspace(pos1.z, pos2.z, ACT_FRAME)
            tempr = pos2.rotation - 360 if pos2.rotation != 0 else 360
            if abs(tempr - pos1.rotation) < abs(pos2.rotation - pos1.rotation):
                pos2.rotation = tempr
            r_list = np.linspace(pos1.rotation, pos2.rotation, ACT_FRAME)
            h_list = np.linspace(pos1.horizon, pos2.horizon, ACT_FRAME)
            for i in range(ACT_FRAME):
                evt = self.ctrler.step(
                    action="Teleport",
                    position=dict(x=x_list[i], y=SAFE_Y, z=z_list[i]),
                    rotation=dict(y=round(r_list[i])),
                    horizon=round(h_list[i])
                )
                assert evt.metadata['lastActionSuccess'],\
                    evt.metadata['errorMessage']
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
        if birdView:
            evt = self.ctrler.step(action='GetMapViewCameraProperties')
            self.ctrler.step(
                action='AddThirdPartyCamera', **evt.metadata['actionReturn'])
        last_p = None
        for p in poses:
            if last_p is not None:
                last_p = AgentPoseState(pose_str=last_p)
            self.Animateframe(
                last_p,
                AgentPoseState(pose_str=p), birdView, smooth)
            last_p = p

    def visualize(
        self,
        scene: str,
        poses: List[str],
        wait: int = 0,
        birdView=False,
        smooth=False
    ):
        assert wait >= 0
        self.get_frames(scene, poses, birdView, smooth)
        print("Press key to start")
        start = True
        for i, p in enumerate(poses):
            if not smooth:
                print("Pose:", p)
                cv2.imshow('vis', self.fp_frames[i])
                if birdView:
                    cv2.imshow("Topview", self.top_frames[i])
                if start:
                    cv2.waitKey(0)
                    start = False
                else:
                    cv2.waitKey(int(wait*1000))
            else:
                if start:
                    cv2.imshow('vis', self.fp_frames[0])
                    if birdView:
                        cv2.imshow("Topview", self.top_frames[0])
                    cv2.waitKey(0)
                    start = False
                print("Pose:", p)
                if i == len(poses) - 1:
                    break
                for j in range(i*ACT_FRAME, (i+1)*ACT_FRAME):
                    cv2.imshow('vis', self.fp_frames[j])
                    if birdView:
                        cv2.imshow("Topview", self.top_frames[j])
                    cv2.waitKey(int(1000/FPS))
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
