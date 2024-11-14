# management/commands/start_attendance.py
from django.core.management.base import BaseCommand
from attendace_app import recognize_attendance

class Command(BaseCommand):
    help = 'Start face recognition for attendance'

    def handle(self, *args, **kwargs):
        recognize_attendance
